using Microsoft.ML.GenAI.Core;
using Microsoft.ML.GenAI.LLaMA;
using Microsoft.ML.Tokenizers;
using static TorchSharp.torch;
using TorchSharp;
using AutoGen.Core;

namespace Local.NET.LLAMA;

internal class Llama
{
    public static async Task RunLlama(string weightFolder, string checkPointName = "model.safetensors.index.json")
    {
        var originalWeightFolder = Path.Combine(weightFolder, "original");

        Console.WriteLine("Loading Llama from huggingface model weight folder");
        var stopWatch = System.Diagnostics.Stopwatch.StartNew();
        stopWatch.Start();

        var pipeline = LoadModel(weightFolder, checkPointName);

        var agent = new LlamaCausalLMAgent(pipeline, "assistant").RegisterPrintMessage();

        var task = "Write a C# program to print the sum of two numbers. Use top-level statement, put code between ```csharp and ```";

        await agent.SendAsync(task);
    }

    public static ICausalLMPipeline<TiktokenTokenizer, LlamaForCausalLM> LoadModel(string weightFolder, string checkPointName = "model.safetensors.index.json")
    {
        var device = "cuda";
        var defaultType = ScalarType.BFloat16;
        torch.manual_seed(1);
        torch.set_default_dtype(defaultType);
        var configName = "config.json";
        var originalWeightFolder = Path.Combine(weightFolder, "original");

        Console.WriteLine("Loading Llama from huggingface model weight folder");
        var tokenizer = LlamaTokenizerHelper.FromPretrained(originalWeightFolder);
        var model = LlamaForCausalLM.FromPretrained(weightFolder, configName, checkPointName: checkPointName, layersOnTargetDevice: -1, quantizeToInt8: false);

        var pipeline = new CausalLMPipeline<TiktokenTokenizer, LlamaForCausalLM>(tokenizer, model, device);

        return pipeline;
    }
}
