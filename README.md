# tfvis for Deno

Visualize Tensorflow.js in Deno.

```typescript
import * as tf from "npm:@tensorflow/tfjs";
import "npm:@tensorflow/tfjs-backend-webgpu";

import * as tfvis from "https://deno.land/x/tfvis/mod.ts";

// See examples/mnist/mnist.ts
// ...

async function showExamples(data) {
  const surface = tfvis.visor().surface({
    name: "Input Data Examples",
    tab: "Input Data",
  });

  const examples = data.nextTestBatch(20);
  const numExamples = examples.xs.shape[0];

  for (let i = 0; i < numExamples; i++) {
    const imageTensor = tf.tidy(() => {
      return examples.xs
        .slice([i, 0], [1, examples.xs.shape[1]])
        .reshape([28, 28, 1]);
    });

    surface.tensor(imageTensor);
  }
}
```

![image](https://github.com/littledivy/tfvis-deno/assets/34997667/0e2639e8-3328-4865-9625-d0c90921aa2b)
