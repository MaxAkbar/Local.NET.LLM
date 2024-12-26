using static TorchSharp.torch;
using TorchSharp;
using Microsoft.ML.GenAI.Mistral;
using Microsoft.ML.Tokenizers;
using Microsoft.ML.GenAI.Core;

namespace Local.NET.Mistral;

public class Util
{
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

    public static CausalLMPipeline<LlamaTokenizer, MistralForCausalLM> LoadMistralModel(string device, string weightFolder, string configName)
    {
        var originalWeightFolder = Path.Combine(weightFolder);
        var tokenizer = MistralTokenizerHelper.FromPretrained(originalWeightFolder);
        var model = MistralForCausalLM.FromPretrained(weightFolder, configName, layersOnTargetDevice: -1);
        var pipeline = new CausalLMPipeline<LlamaTokenizer, MistralForCausalLM>(tokenizer, model, device);

        return pipeline;
    }
}
