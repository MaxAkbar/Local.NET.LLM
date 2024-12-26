using AutoGen.Core;
using System.Reflection;

namespace Local.NET.Mistral;

public class FunctionCall
{

    //public FunctionContract GetFunctionContract(MethodInfo methodInfo)
    //{
    //    var parameters = methodInfo.GetParameters()
    //        .Select(p => new FunctionParameterContract
    //        {
    //            Name = p.Name,
    //            Description = $"{p.Name}. type is {p.ParameterType.Name}",
    //            ParameterType = p.ParameterType,
    //            IsRequired = !p.IsOptional,
    //        })
    //        .ToArray();

    //    return new FunctionContract
    //    {
    //        Namespace = methodInfo.DeclaringType!.Namespace,
    //        ClassName = methodInfo.DeclaringType!.Name,
    //        Name = methodInfo!.Name,
    //        Description = methodInfo.GetCustomAttribute<FunctionAttribute>()?.Description ?? "No description",
    //        ReturnType = methodInfo.ReturnType,
    //        Parameters = parameters,
    //    };
    //}
}