using Local.NET.LLAMA;
using Microsoft.ML.GenAI.Samples.Llama;

await Llama.RunLlama(@"C:\Users\maxim\source\repos\models\meta\Llama-3.2-1B-Instruct", "model.safetensors");
await LlamaTraining.Train(@"C:\Users\maxim\source\repos\models\meta\Llama-3.2-1B-Instruct", "model.safetensors");
await Llama.RunLlama(@"C:\Users\maxim\source\repos\models\meta\Llama-3.2-1B-Instruct", "contoso-llama-3.1-1b.safetensors");