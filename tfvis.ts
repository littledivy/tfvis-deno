/*
 * Realtime Tensorflow visualization for Deno.
 *
 * Copyright (c) 2023 Divy Srivastava
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
import {
  EventType,
  PixelFormat,
  Rect,
  TextureAccess,
  WindowBuilder,
} from "https://deno.land/x/sdl2/mod.ts";

const eventLoop = Symbol("eventLoop");

class LineGraphCommand {
  static submit(values, surface) {
    const command = Object.create(LineGraphCommand.prototype);
    command.values = values;
    return command;
  }

  async draw(canvas, offset = 0) {
    const values = this.values;
    if (values.length < 2) {
      return;
    }
    const padding = 10;
    const width = 640 - padding;
    const height = 480 - padding;

    const min = Math.min(...values);
    const max = Math.max(...values);
    const range = max - min;
    const step = width / values.length;

    // Draw the axes.
    canvas.drawLine(padding, padding, padding, height);
    canvas.drawLine(padding, height, width, height);

    for (let i = 0; i < values.length; i++) {
      const x = padding + i * step;
      const y = height - (values[i] - min) / range * height;
      // Draw a line to the next point.
      if (i < values.length - 1) {
        const nextX = padding + (i + 1) * step;
        const nextY = height - (values[i + 1] - min) / range * height;
        if (Number.isNaN(nextY)) continue;
        canvas.drawLine(
          Math.round(x),
          Math.round(y),
          Math.round(nextX),
          Math.round(nextY),
        );
      }
    }
  }
}

class TensorCommand {
  static async submit(t, surface) {
    const command = Object.create(TensorCommand.prototype);
    command.t = t;

    const values = await t.data();
    const [height, width] = t.shape.slice(0, 2);

    const multiplier = t.dtype === "float32" ? 255 : 1;
    const bytes = new Uint8Array(width * height * 4);

    for (let i = 0; i < values.length; i++) {
      const value = values[i] * multiplier;
      bytes[i * 4 + 0] = value;
      bytes[i * 4 + 1] = value;
      bytes[i * 4 + 2] = value;
      bytes[i * 4 + 3] = 255;
    }

    const textureCreator = surface.textureCreator();
    const texture = textureCreator.createTexture(
      PixelFormat.Unknown,
      TextureAccess.Streaming,
      width,
      height,
    );

    texture.update(bytes, width * 4);
    command.texture = texture;

    return command;
  }

  async draw(surface, offset = 0) {
    const padding = 10;
    const spacing = 2;
    const target = new Rect(
      padding + (this.t.shape[1] + spacing) * offset,
      padding,
      this.t.shape[1],
      this.t.shape[0],
    );
    surface.copy(this.texture, null, target);
  }
}

class Surface {
  #commands = [];

  constructor(window) {
    this.window = window;

    const canvas = window.canvas();
    canvas.setDrawColor(255, 255, 255, 255);
    canvas.clear();
    canvas.present();

    this.canvas = canvas;
  }

  async tensor(t) {
    this.#commands.push(await TensorCommand.submit(t, this.canvas));
  }

  lineGraph(values) {
    this.#commands.push(LineGraphCommand.submit(values, this.canvas));
  }

  async [eventLoop]() {
    const canvas = this.canvas;
    const { value: event } = this.window.events().next();
    if (event.type === EventType.Quit) {
      return;
    }

    if (event.type === EventType.Draw) {
      canvas.clear();
      canvas.setDrawColor(0, 0, 0, 255);
      let offset = 0;
      for (const command of this.#commands) {
        await command.draw(canvas, offset++);
      }
      canvas.present();
      canvas.setDrawColor(255, 255, 255, 255);
    }

    setTimeout(() => this[eventLoop](), 0);
  }
}

class Visor {
  surface({ name, tab, styles }) {
    const window = new WindowBuilder(name, 640, 480).build();

    const surface = new Surface(window);
    surface[eventLoop]();

    return surface;
  }
}

function visor() {
  return new Visor();
}

const show = {
  fitCallbacks: function (container, opts) {
    const callbackNames = opts.callbacks || ["onEpochEnd", "onBatchEnd"];

    const surface = visor().surface(container);
    function makeCallback(callbackName) {
      const values = [];
      surface.lineGraph(values);
      return async function (_, logs) {
        // TODO
      };
    }

    const callbacks = {};
    callbackNames.forEach((callbackName) => {
      callbacks[callbackName] = makeCallback(callbackName);
    });
    return callbacks;
  },
};

const metrics = {};

const render = {};

export { metrics, render, show, visor };
