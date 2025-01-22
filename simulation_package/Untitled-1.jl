# cd("/Users/alihashim/Desktop/Online_Academic_Submissions/poec_thesis/simulation_package")
# include("SimulationPackage.jl")

using DataFrames
using DataStructures

"""
    is_structlike(x)

Returns true if `x` looks like a struct or a NamedTuple that we can iterate via `fieldnames`.
"""
function is_structlike(x)
    # NamedTuple is easy to check
    if x isa NamedTuple
        return true
    end
    # For most concrete structs you define, fieldnames(...) is non-empty
    # or at least recognized. (Be sure x is not nothing or missing before calling!)
    T = typeof(x)
    # skip "primitive" (like Int, Float64, etc.) or abstract types
    if isprimitivetype(T) || isabstracttype(T)
        return false
    end
    # Heuristic: struct if it has at least one field name
    return !isempty(fieldnames(T))
end

"""
    flatten_struct!(acc, obj; prefix = "")

Recursively flattens `obj` into the dictionary `acc`, with column names formed
from `prefix` plus the field name. Skips any field whose name contains "sd" or "p_val".
Expands numeric vectors. Stores bool/int/float fields as single values.
Handles `missing` and `nothing` by storing them as single columns (unless prefix is empty).
"""
function flatten_struct!(acc::AbstractDict{String,Any}, obj; prefix::String="")
    # 1) If the object is `missing`, store it if we have a prefix, then return
    if obj === missing
        if !isempty(prefix)
            acc[prefix] = missing
        end
        return
    end

    # 2) If the object is `nothing`, we can either store it or skip:
    if obj === nothing
        if !isempty(prefix)
            acc[prefix] = nothing
        end
        return
    end

    # 3) If the object is a Bool, Int, Float, etc. store under prefix (if prefix != "")
    if obj isa Number || obj isa Bool
        if !isempty(prefix)
            acc[prefix] = obj
        end
        return
    end

    # 4) If the object is a numeric Vector, expand each element into prefix_1, prefix_2, ...
    if obj isa AbstractVector{<:Number}
        if !isempty(prefix)
            for (i, val) in enumerate(obj)
                acc["$(prefix)_$(i)"] = val
            end
        end
        return
    end

    # 5) If it's some other vector (non-numeric), decide how to handle. Example: skip or flatten recursively.
    #    For demonstration, let's skip them or store as-is. 
    if obj isa AbstractVector
        # If you want to store them as is:
        if !isempty(prefix)
            acc[prefix] = obj
        end
        return
    end

    # 6) If it's struct-like, we attempt to flatten its fields
    if is_structlike(obj)
        # flatten each field
        for fn in fieldnames(typeof(obj))
            fn_str = String(fn)
            # skip if the name has "sd" or "p_val"
            if occursin("sd", fn_str) || occursin("p_val", fn_str)
                continue
            end
            fieldval = getfield(obj, fn)
            # build next prefix
            new_prefix = isempty(prefix) ? lowercase(fn_str) : string(prefix, "_", lowercase(fn_str))
            flatten_struct!(acc, fieldval; prefix=new_prefix)
        end
        return
    end

    # 7) If none of these matched, we can store it as-is if we have a prefix
    if !isempty(prefix)
        acc[prefix] = obj
    end
end


"""
    flatten_into_dict(result_structs...; remove_substring = "")

Flattens one or more result objects into a single dictionary.
If `remove_substring` is provided, it removes that substring from keys
before converting them to Symbols.
"""
function flatten_into_dict(result_structs...; remove_substring="")
    acc = Dict{String,Any}()
    for rs in result_structs
        # Use the type's name as a top-level prefix, e.g., "spatialautocorrelationmeasurement"
        type_prefix = lowercase(string(typeof(rs).name.wrapper))

        flatten_struct!(acc, rs; prefix=type_prefix)
    end

    # Step 1: Remove the specified substring from all keys
    if !isempty(remove_substring)
        acc = Dict(replace(key, remove_substring => "") => value for (key, value) in acc)
    end

    return acc
end

# Flatten and create a DataFrame:
measurements = flatten_into_dict(
    spatial_corr_measurements, prop_eval_metrics, majoritarian_eval_metrics, tangian_indices;
    remove_substring="main.hashimpoecthesissimulationpackage."
)

ordered_result = OrderedDict(
    RAW_NAME_TO_NICE_NAME3[key] => measurements[key] for key in keys(RAW_NAME_TO_NICE_NAME3)
)

df = DataFrame(mydict)
