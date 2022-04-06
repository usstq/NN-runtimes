#!/usr/bin/python3

# import openvino.runtime as ov
from more_itertools import value_chain
import numpy as np
from graphviz import Digraph, Source
import ctypes, re

# print Model in readable text
def generate_str(model, show_rt_info = False):
    out2name = {}
    nameid = 1
    simpleconst_node2vstr = {}
    ilist = [i.get_node().get_name() for i in model.inputs]
    result = []
    def get_rt_info(n):
        return {k:str(v) for k,v in n.get_rt_info().items()}
    result.append("model({}):".format(",".join(ilist)))

    for k, v in model.get_rt_info().items():
        result.append("  {}={}".format(k,v))
    for n in model.get_ordered_ops():
        # collect output and also allocate output names
        rt_info = get_rt_info(n)
        if "reginfo" in rt_info or "effectiveAddress" in rt_info:
            if "reginfo" in rt_info:
                varname = "vmm{}".format(rt_info["reginfo"])
            else:
                varname = "t{}".format(nameid)
                nameid += 1
            str_output = varname
            args = []
            if "effectiveAddress" in rt_info:
                args.append("[r{}]".format(rt_info["effectiveAddress"]))
            for i in n.inputs():
                r2 = get_rt_info(i.get_source_output().get_node())
                if "reginfo" in r2:
                    args.append("vmm{}".format(r2["reginfo"]))
                else:
                    args.append(out2name[i.get_source_output()])

            for k, out in enumerate(n.outputs()):
                out2name[out] = varname if num_out == 1 else "{}[{}]".format(varname, k)
        else:
            out_types = []
            varname = "t{}".format(nameid)
            nameid += 1
            num_out = len(n.outputs())
            for k, out in enumerate(n.outputs()):
                out_types.append("Tensor<{}x{}>".format(
                                    "x".join([str(s) for s in out.get_shape()]),
                                    out.get_element_type().get_type_name()))
                out2name[out] = varname if num_out == 1 else "{}[{}]".format(varname, k)

            #out_types
            str_out_types = out_types[0] if len(out_types)==1 else "tuple({})".format(",".join(out_types))
            str_output = "{} {}".format(str_out_types, varname)
        
            # collect source output names of corresponding inputs
            args = []
            for i in n.inputs():
                o = i.get_source_output()
                if o in simpleconst_node2vstr:
                    args.append(simpleconst_node2vstr[o])
                else:
                    args.append(out2name[o])

        # generate psuedo code
        type_name = n.get_type_name()
        friendly_name = n.get_friendly_name()
        rt_info = n.get_rt_info()
        if type_name == "ExecutionNode" and "layerType" in rt_info:
            type_name = str(rt_info["layerType"])
        attrs = ["{}={}".format(k, v) for k,v in n.get_attributes().items()]
        rtinfo = ["{}={}".format(k, v) for k,v in rt_info.items()]

        comment = friendly_name
        comment = "" if len(comment)==0 else "   # {}".format(comment)
        if type_name.startswith("Constant"):
            vstr = n.get_value_strings()
            if len(vstr) <= 8:
                simpleconst_node2vstr[n.outputs()[0]] = "[{}]".format(",".join(vstr))
            else:
                result.append("    {} = {}([{}]) {}".format(
                                    str_output,
                                    type_name,
                                    ",".join(vstr[:16]) + (",..." if len(vstr)>16 else ""),
                                    comment))
        else:
            result.append("    {} = {}({}{}) {}".format(
                        str_output,
                        type_name,
                        ",".join(args),
                        "" if len(attrs) == 0 else ("," if len(args)>0 else "") + (",".join(attrs)),
                        comment ))
        if (show_rt_info and rtinfo):
            result.append("\t\t\t#rt_info:\n\t\t\t#\t{}\n".format("\n\t\t\t#\t".join(rtinfo)))

    olist = [out2name[i] for i in model.outputs]
    result.append("    return {}".format(",".join(olist)))
    return "\n".join(result)

def print_model(model, show_rt_info = False):
    print(generate_str(model, show_rt_info))

def generate_graph(model, fontsize=12, graph_name="", detailed_label=False):
    # create all nodes before edges
    g = Digraph(graph_name, graph_attr={"outputorder":"edgesfirst"})
    node2name = {}
    name2node = {}
    data_map = {}
    data_color = {}
    precision2ctype = {
        "I8":ctypes.c_int8,
        "U8":ctypes.c_uint8,
        "I32": ctypes.c_int32,
        "FP32":ctypes.c_float
    }

    def gen_rand_color():
        if not hasattr(gen_rand_color, "color_hue"):
            gen_rand_color.color_hue = 0
        gen_rand_color.color_hue = (gen_rand_color.color_hue + 5/8) % 1
        return "{:.3f} 1 0.7".format(gen_rand_color.color_hue)

    def strings2label(strings, nlimit = 20, line_limit = 1):
        r = ""
        line = 0
        prev_cnt = 0
        for s in strings:
            if len(r) + len(s) - prev_cnt > nlimit:
                r += "\\n"
                prev_cnt = len(r)
                line += 1
                if line >= line_limit:
                    r += "..."
                    break
            r += s + ","
        return r.rstrip(",")

    op2color = {"Parameter":"gold", "Result":"deeppink", "Constant":"gray", "Const":"gray"}

    inode2index = {input.node:k for k,input in enumerate(model.inputs)}


    def name_normalize(n):
        name = n.get_friendly_name()
        # add input id if it's input node of the model
        if n in inode2index:
            name += "_#{}".format(inode2index[n])
        name = name.replace("<","(").replace(">",")").replace(":","_")
        if len(graph_name):
            return '{}'.format(graph_name, name)
        return '{}'.format(name)

    # statistics on execTime
    execTimeMcs_total = 0
    execTimeMcs_by_type = {}
    execTimeMcs_by_node = {}
    for n in model.get_ordered_ops():
        friendly_name = name_normalize(n)
        rt_info = n.get_rt_info()
        type_name = n.get_type_name()
        if type_name == "ExecutionNode" and "layerType" in rt_info:
            type_name = str(rt_info["layerType"])
        if "primitiveType" in rt_info:
            type_name += "({})".format(rt_info["primitiveType"])
        execTimeMcs = 0
        if ("execTimeMcs" in rt_info):
            execTimeMcs = rt_info["execTimeMcs"]
            try:
                execTimeMcs = int(execTimeMcs)
            except:
                execTimeMcs = 0
        execTimeMcs_by_node[n] = execTimeMcs
        execTimeMcs_total += execTimeMcs
        if type_name in execTimeMcs_by_type:
            execTimeMcs_by_type[type_name] += execTimeMcs
        else:
            execTimeMcs_by_type[type_name] = execTimeMcs

    if execTimeMcs_total > 0:
        num_limit = 10
        sort_execTimeMcs_by_type = []
        acc_percentage = 0
        for (type_name, t) in sorted(execTimeMcs_by_type.items(), key=lambda x: x[1], reverse=True):
            percentage = t*100/execTimeMcs_total
            acc_percentage += percentage
            sort_execTimeMcs_by_type.append("{:>10}  {:.1f}%  accumulated:{:.1f}%".format(type_name, percentage, acc_percentage))
            if acc_percentage >= 90 and len(sort_execTimeMcs_by_type) >= num_limit:
                break
        
        sort_execTimeMcs_by_name = []
        acc_percentage = 0
        for (n, t) in sorted(execTimeMcs_by_node.items(), key=lambda x: x[1], reverse=True):
            friendly_name = name_normalize(n)
            type_name = n.get_type_name()
            rt_info = n.get_rt_info()
            if type_name == "ExecutionNode" and "layerType" in rt_info:
                type_name = str(rt_info["layerType"])
            percentage = t*100/execTimeMcs_total
            acc_percentage += percentage
            sort_execTimeMcs_by_name.append("{:>10}({})  {:.1f}%   accumulated:{:.1f}%".format(friendly_name, type_name, percentage, acc_percentage))
            if acc_percentage >= 90 and len(sort_execTimeMcs_by_name) >= num_limit:
                break
        
        kwargs = {"shape":'box',
                "style":'filled',
                "fillcolor":"gold",
                "fontsize":str(fontsize + 2),
                "margin":"0,0","width":"0","height":"0",
                "tooltip":"\n".join(sort_execTimeMcs_by_type)}
        g.node(name="ProfileSummary_ByType",
                label="ProfileSummary\\nByType",
                **kwargs)

        kwargs = {"shape":'box',
                "style":'filled',
                "fillcolor":"gold",
                "fontsize":str(fontsize + 2),
                "margin":"0,0","width":"0","height":"0",
                "tooltip":"\n".join(sort_execTimeMcs_by_name)}
        g.node(name="ProfileSummary_ByName",
                label="ProfileSummary\\nByName",
                **kwargs)

    for nindex, n in enumerate(model.get_ordered_ops()):
        friendly_name = name_normalize(n)
        rt_info = n.get_rt_info()
        type_name = n.get_type_name()
        if friendly_name in name2node:
            print("WARNNING:  {} (type {}) already exist as {}, skipped!".format(
                friendly_name, type_name,
                name2node[friendly_name].get_type_name()))
            continue
        
        # ExecutionNode is fake wrapper of runtime node
        # and this type name gives less information than friendly_name
        if type_name == "ExecutionNode" and "layerType" in rt_info:
            type_name = str(rt_info["layerType"])

        attrs = ["{}={}".format(k, v) for k,v in n.get_attributes().items()]
        def rtinfo2string(k, v):
            if k == "originalLayersNames":
                v = "\n    " + str(v).replace(",","\n    ")
            return "{}={}".format(k, v)
        rtinfo = [rtinfo2string(k, v) for k,v in rt_info.items()]

        # originalLayersNames gives mapping between runtime nodes and orginal nodes
        fsize = fontsize
        if type_name == "Constant":
            vstr = n.get_value_strings()
            label = strings2label(vstr)
            fsize = fontsize - 2
        elif "fusedTypes" in rt_info:
            label = "{" + rt_info['fusedTypes'].replace(",","|") +"}"
        else:
            label = "{}\\n({}{})".format(type_name, friendly_name[:20], "..." if len(friendly_name)>20 else "")

        allinfo = "{} / #{}".format(friendly_name, nindex)

        if (execTimeMcs_by_node[n] > 0):
            allinfo += "\n----execTime of the node {:.2f}%".format(execTimeMcs_by_node[n]*100/execTimeMcs_total)
        if (attrs):
            allinfo += "\n----attributes----\n{}".format("\n".join(attrs))
        if (rt_info):
            allinfo += "\n----rt_info----\n{}".format("\n".join(rtinfo))

        #if type_name.startswith("Constant"):
        #    allinfo += "\n----values----\n{}".format(",".join(n.get_value_strings()[:32]))

        color = op2color[type_name] if type_name in op2color else "cyan"
        if type_name == "Subgraph":
            submodel = rt_info["body"]
            allinfo += "\n----model-----\n{}".format(generate_str(submodel))
            data_map[friendly_name] = submodel

        if detailed_label:
            label = allinfo.replace("\n", "\\n")
            allinfo = label.replace("\\n", "\n")
        kwargs = {"shape":'Mrecord',
              "style":'filled,rounded',
               "fillcolor":color,
               "fontsize":str(fsize),
               "margin":"0,0","width":"0","height":"0",
               "tooltip":allinfo}
        g.node(name=friendly_name,
               label=label,
               **kwargs)
        assert(friendly_name not in name2node) # make sure the name is uinque
        name2node[friendly_name] = n
        node2name[n] = friendly_name
    
    # generate color table for in-place mem
    for n in model.get_ordered_ops():
        for i in n.inputs():
            mem_rt_info = i.get_source_output().get_rt_info()
            if "Data" in mem_rt_info:
                Data = mem_rt_info["Data"]
                if not Data in data_color:
                    # single non-inplace color is black
                    data_color[Data] = "black"
                elif data_color[Data] == "black":
                    # replace in-place color with non-black
                    data_color[Data] = gen_rand_color()

    max_act_sz = 0
    for n in model.get_ordered_ops():
        for i in n.inputs():
            act_sz = np.prod(np.array(i.get_shape()))
            if (max_act_sz < act_sz):
                max_act_sz = act_sz

    for n in model.get_ordered_ops():
        for i in n.inputs():
            src_out = i.get_source_output()
            tail_name = name_normalize(src_out.get_node())
            head_name = name_normalize(n)

            if (len(src_out.get_target_inputs()) > 4 or len(src_out.get_node().outputs()) > 4):
                found_ki = False
                for ki, si in enumerate(src_out.get_target_inputs()):
                    if si.get_node() is n:
                        found_ki = True
                        break
                assert(found_ki)
                tail_name += ".out{}.{}".format(src_out.get_index(), ki)

            act_sz = np.prod(np.array(i.get_shape()))
            str_shape = ",".join([str(s) for s in i.get_shape()])
            str_ele_type = i.get_element_type().get_type_name()
            src_rt_info = i.get_source_output().get_node().get_rt_info()
            mem_rt_info = i.get_source_output().get_rt_info()

            label = '[{}]'.format(str_shape)
            layout_fmt = None
            if "Format" in mem_rt_info:
                layout_fmt = mem_rt_info["Format"]
            elif "outputLayouts" in src_rt_info:
                layout_fmt = src_rt_info["outputLayouts"]

            if layout_fmt not in ("a","ab","abc","abcd","abcde","abcdef",None):
                label += "\n" + layout_fmt

            precision = None
            if "Precision" in mem_rt_info:
                precision = mem_rt_info["Precision"]
            elif "outputPrecisions" in src_rt_info:
                precision = src_rt_info["outputPrecisions"]
            else:
                precision = str_ele_type
            
            if precision not in ("FP32","float","float32",None):
                label += "\n" + precision

            color = "black"
            if "Data" in mem_rt_info:
                Data = mem_rt_info["Data"]
                #label += "\n0x{:X}".format(Data)
                try:
                    # build a numpy array and return
                    p=ctypes.c_void_p(Data)
                    c_type = precision2ctype[mem_rt_info["Precision"]]
                    pf = ctypes.cast(p, ctypes.POINTER(c_type))
                    cnt = mem_rt_info["MaxMemSize"]//ctypes.sizeof(c_type)
                    base_array = np.ctypeslib.as_array(pf, shape=(cnt,))
                    part_array = base_array[mem_rt_info["OffsetPadding"]:]
                    BlockDims = mem_rt_info["BlockDims"]
                    OffsetPaddingToData = mem_rt_info["OffsetPaddingToData"]
                    Strides = mem_rt_info["Strides"]

                    total_shape = np.array(BlockDims) + np.array(OffsetPaddingToData)
                    total_cnt = np.prod(total_shape)
                    new_array = part_array[:total_cnt].reshape(total_shape)

                    nd_strides = np.array(new_array.strides)//ctypes.sizeof(c_type)
                    if (nd_strides != np.array(Strides)).any():
                        # TODO new_array = part_array.reshape(np.array(Strides))
                        label += "\n(strided)"

                    color = data_color[Data]
                    if not Data in data_map:
                        data_map[Data] = []
                    data_map[Data].append(new_array)
                except Exception as e:
                    print("edge '{}->{}' with Data but failed to parse:\n{}".format(
                        tail_name, head_name, e
                    ))
                    raise e

            labeltooltip = []
            for k,v in mem_rt_info.items():
                if k == "Data" or k == "Ptr":
                    value = "0x{:X}".format(mem_rt_info[k])
                else:
                    value = str(v)
                labeltooltip.append("{}={}".format(k, value))
            penwidth = act_sz*4.5/max_act_sz + 0.5
            g.edge(
                tail_name,
                head_name,
                label=label,
                edgetooltip="{}:{}->{}:{}".format(tail_name, i.get_source_output().get_index(), head_name, i.get_index()),
                labeltooltip="\n".join(labeltooltip),
                headURL="head",
                headtooltip="headtooltip",
                tailtooltip="tailtooltip",
                color=color,
                penwidth = "{:.3f}".format(penwidth),
                fontsize=str(fontsize*8//10))
    return g, data_map

def visualize_model(model, fontsize=12, filename=None, detailed_label=False):
    g, data_map = generate_graph(model, fontsize, detailed_label=detailed_label)
    graph_src = Source(g.source, format="svg")
    if filename:
        svg = graph_src.pipe().decode('utf-8')
        if filename.endswith(".html"):
            import dot_svg_html
            output_src = dot_svg_html.dot_to_html(svg)
        else:
            output_src = svg
        htmlfile = open(filename,'w')
        htmlfile.write(output_src)
        htmlfile.close()
        return
    return graph_src, data_map
# for measuring CPU usage in separate process
import psutil, time
from multiprocessing import Process, Pipe
def worker_process(conn, percpu):
    cpu_usage = []
    while (not conn.poll()):
        time.sleep(0.1)
        cpu_usage.append(psutil.cpu_percent(percpu=percpu))
    conn.recv()
    conn.send(cpu_usage)
    conn.close()

class CPUUsage:
    def __init__(self) -> None:
        self.parent_conn, self.child_conn = Pipe()

    def start(self, percpu=False):
        self.p = Process(target=worker_process, args=(self.child_conn,percpu))
        self.p.start()

    def end(self):
        self.parent_conn.send("finish")
        cpu_usage = self.parent_conn.recv()
        self.p.join()
        return cpu_usage

# helper to dump model
from openvino.runtime.passes import Manager
import openvino.runtime as ov

def serialize_model(self, model_path):
    weight_path = model_path[:model_path.find(".xml")] + ".bin"
    pass_manager = Manager()
    pass_manager.register_pass("Serialize", model_path, weight_path)
    pass_manager.run_passes(self)
    return model_path, weight_path

# https://stackoverflow.com/questions/47797661/python-types-methodtype
# add serialize method
ov.Model.serialize = serialize_model
ov.Model.print = print_model
ov.Model.visualize = visualize_model


from openvino.runtime.utils.types import get_dtype
def fill_tensors_with_random(input, alpha=0):
    dtype = get_dtype(input.get_element_type())
    rand_min, rand_max = (0, 1) if dtype == np.bool else (np.iinfo(np.uint8).min, np.iinfo(np.uint8).max)
    # np.random.uniform excludes high: add 1 to have it generated
    if np.dtype(dtype).kind in ['i', 'u', 'b']:
        rand_max += 1
    rs = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(0)))
    shape = input.get_shape()
    a = rs.uniform(rand_min, rand_max, list(shape)).astype(dtype)
    return ov.Tensor(a)

def test_infer_queue(compiled_model, num_request, num_infer, time_limit=60):
    infer_queue = ov.AsyncInferQueue(compiled_model, num_request)

    latency_list = []
    prof_list = []
    def callback(request, userdata):
        latency_list.append(request.latency)
        prof_list.append(request.profiling_info)

    infer_queue.set_callback(callback)

    all_input = {}
    for port, input in enumerate(compiled_model.inputs):
        print("input[{}]  {:<10} {} {}".format(port, input.get_any_name(), input.get_element_type(), input.get_shape()))
        all_input[port] = fill_tensors_with_random(input)

    for i in range(num_request):
        infer_queue.start_async(all_input, userdata=i)

    t0 = time.time()
    for i in range(num_infer):
        wtime = time.time() - t0
        if time_limit and (wtime > time_limit):
            break
        infer_queue.start_async(None, userdata=i)
    infer_queue.wait_all()
    fps = i/wtime
    return latency_list, prof_list, fps, wtime

if __name__ == "__main__":

    #test222()
    #test_visualize()

    import openvino.runtime as ov
    import numpy as np
    import sys, os

    core = ov.Core()
    model_path = sys.argv[1]
    model = core.read_model(model_path)
    #model.visualize(filename="{}.dot".format(model_path))
    #model.print()
    if "OPT_LINENUM" in os.environ:
        OPT_LINENUM = os.environ["OPT_LINENUM"]
    else:
        OPT_LINENUM = ""

    device = "CPU"
    NUM_STREAMS = 1
    INFERENCE_NUM_THREADS = 1

    dev_prop = {"PERF_COUNT": "YES",
                "AFFINITY": "CORE",
                "PERFORMANCE_HINT_NUM_REQUESTS":0,
                "PERFORMANCE_HINT":""}
    if (NUM_STREAMS):
        dev_prop["NUM_STREAMS"] = NUM_STREAMS
    if (INFERENCE_NUM_THREADS):
        dev_prop["INFERENCE_NUM_THREADS"] = INFERENCE_NUM_THREADS

    core.set_property(device, dev_prop)

    if False:
        dest_file = filename="{}_org.html".format(model_path)
        print("saving {} ...".format(dest_file))
        model.visualize(filename=dest_file)
        print("{} is saved!".format(dest_file))

    #model.reshape(ov.PartialShape([2,512]))
    compiled_model = core.compile_model(model, "CPU")

    def test_infer():
        req = compiled_model.create_infer_request()
        for i in range(1):
            all_input = {}
            for input in compiled_model.inputs:
                print("{:<10} {} {}".format(input.get_any_name(), input.get_element_type(), input.get_shape()))
                all_input[input] = fill_tensors_with_random(input)
            req.infer(inputs=all_input)
            print(req.output_tensors)
            #from PIL import Image
            #im = Image.fromarray(req.output_tensors[0].data.squeeze().astype(np.uint8))
            #im.save("your_file_{}.png".format(a))
            sys.exit(0) 
    #test_infer()

    latency_list, prof_list, fps, wtime = test_infer_queue(compiled_model, 2, 20000, time_limit=10)
    print(f"test_infer_queue FPS:{fps:.1f}")

    dest_file = filename="{}_{}_{}.html".format(model_path, device, OPT_LINENUM)
    print("saving {} ...".format(dest_file))
    compiled_model.get_runtime_model().visualize(filename=dest_file)
    print("{} is saved!".format(dest_file))
