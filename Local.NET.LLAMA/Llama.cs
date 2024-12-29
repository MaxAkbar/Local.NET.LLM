using Microsoft.ML.GenAI.Core;
using Microsoft.ML.GenAI.LLaMA;
using Microsoft.ML.Tokenizers;
using static TorchSharp.torch;
using TorchSharp;
using AutoGen.Core;
using static TorchSharp.torch.nn;
using System.Text;
using Microsoft.Extensions.AI;

namespace Local.NET.LLAMA;

internal class Llama
{
    private static LlamaFunctionCall instance = new();

    private static async Task UseAgentAsync(LlamaCausalLMAgent agent)
    {
        Console.WriteLine("You are now chating with Llama. To exit type '/exit'. To restart the chat just enter.");

        var messages = new List<TextMessage>();

        while (true)
        {
            Console.Write("User> ");

            var userQuery = Console.ReadLine();
            var llamaResponse = new StringBuilder();
            var generateReplyOptions = new GenerateReplyOptions
            {
                Temperature = 0f,
                MaxToken = 1024
            };

            if (string.IsNullOrEmpty(userQuery))
            {
                Console.Clear();

                await UseAgentAsync(agent);
            }

            if (userQuery!.Contains("/exit"))
            {
                break;
            }

            messages.Add(new TextMessage(new TextMessageUpdate(Role.User, userQuery)));
            Console.Write("Llama> ");

            await foreach (var resp in agent.GenerateStreamingReplyAsync(messages, generateReplyOptions))
            {
                llamaResponse.Append(resp.GetContent());
                Console.Write(resp.GetContent());
            }

            Console.WriteLine();
            messages.Add(new TextMessage(new TextMessageUpdate(Role.Assistant, llamaResponse.ToString())));
        }

        Console.WriteLine("Bye!");
    }

    public static async Task RunLlama(string weightFolder, string checkPointName = "model.safetensors.index.json")
    {
        var pipeline = LoadModel(weightFolder, checkPointName);
        var agent = new LlamaCausalLMAgent(pipeline, "assistant");

        await UseAgentAsync(agent);
    }

    public static ICausalLMPipeline<TiktokenTokenizer, LlamaForCausalLM> LoadModel(string weightFolder, string checkPointName = "model.safetensors.index.json")
    {
        var (device, defaultType) = InitializeTorch();
        var configName = "config.json";
        var originalWeightFolder = Path.Combine(weightFolder, "original");

        Console.WriteLine("Loading Llama...");

        var tokenizer = LlamaTokenizerHelper.FromPretrained(originalWeightFolder);
        var model = LlamaForCausalLM.FromPretrained(weightFolder, configName, checkPointName: checkPointName, layersOnTargetDevice: -1, quantizeToInt8: false);

        var pipeline = new CausalLMPipeline<TiktokenTokenizer, LlamaForCausalLM>(tokenizer, model, device);

        return pipeline;
    }

    public static (string device, ScalarType defaultType) InitializeTorch()
    {
        var device = "cuda";
        var defaultType = ScalarType.BFloat16;

        if (device == "cuda")
        {
            torch.InitializeDeviceType(DeviceType.CUDA);
        }

        torch.manual_seed(1);
        torch.set_default_dtype(defaultType);

        return (device, defaultType);
    }
}
