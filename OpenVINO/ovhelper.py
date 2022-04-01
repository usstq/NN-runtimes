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
        if (execTimeMcs_by_type[type_name] > 0):
            allinfo += "\n----execTime of the type {:.2f}%".format(execTimeMcs_by_type[type_name]*100/execTimeMcs_total)

        if (attrs):
            allinfo += "\n----attributes----\n{}".format("\n".join(attrs))
        if (rt_info):
            allinfo += "\n----rt_info----\n{}".format("\n".join(rtinfo))

        if type_name.startswith("Constant"):
            allinfo += "\n----values----\n{}".format(",".join(n.get_value_strings()[:32]))

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
            tail_name = name_normalize(i.get_source_output().get_node())
            head_name = name_normalize(n)
            
            act_sz = np.prod(np.array(i.get_shape()))
            str_shape = ",".join([str(s) for s in i.get_shape()])
            str_ele_type = i.get_element_type().get_type_name()
            src_rt_info = i.get_source_output().get_node().get_rt_info()
            mem_rt_info = i.get_source_output().get_rt_info()

            label = '[{}]'.format(str_shape)
            if "Format" in mem_rt_info:
                label += "\n" + mem_rt_info["Format"]
            elif "outputLayouts" in src_rt_info:
                label += "\n" + src_rt_info["outputLayouts"]

            if "Precision" in mem_rt_info:
                label += "\n" + mem_rt_info["Precision"]
            elif "outputPrecisions" in src_rt_info:
                label += "\n" + src_rt_info["outputPrecisions"]
            else:
                label += "\n" + str_ele_type

            color = "black"
            if "Data" in mem_rt_info:
                Data = mem_rt_info["Data"]
                label += "\n0x{:X}".format(Data)
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

html_prefix ='''<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN"
 "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
<!-- Generated by graphviz version 2.40.1 (20161225.0304)
 -->
<html>
<head>
    <style>
        .infobar {
            position: absolute;
            z-index: 100;
            background-color: #ffffff;
            overflow: auto;
            border: 1px solid black;
        }

        pre {
            font-family: ui-monospace, SFMono-Regular, SF Mono, Menlo, Consolas, Liberation Mono, monospace;
            font-size: 12px;
            margin: 5px;
        }

        .grabbable {
            position: absolute;
            cursor: move;
            cursor: grab;
            cursor: -moz-grab;
            cursor: -webkit-grab;
        }

        .search_li {
            cursor: pointer;
        }
        .search_li:hover {
            background-color: #ffffff;
        }
    </style>
</head>
<body>
<pre id="SearchPointer" style="position:absolute;z-index: 100;font-size:32px;color: #ff0000;pointer-events: none;">&starf;</pre>
<div id="floatbar" style="position:fixed;top:20px;right: 10px;z-index: 100;background-color: #92cff3;border: 1px solid black;padding:5px;">
    &#x1F50D;&#8239;<input type="text" id="SearchKey" placeholder="search here">&#8239;&crarr;<br>
  <div style="max-height: 450px;overflow-y: auto;">
  <ui id="SearchList">
  </ui>
  </div>
</div>
'''
html_surfix='''
<div id="infobar" class="infobar">
    <pre id="infopre">
</pre>
</div>

<script type="text/javascript">
    infobar = document.querySelector("#infobar")
    infopre = document.querySelector("#infopre")
    SearchKey = document.querySelector("#SearchKey")
    Search = document.querySelector("#Search")
    SearchPointer = document.querySelector("#SearchPointer")
    SearchList = document.querySelector("#SearchList")
    SearchKey.onkeyup = function (ev){
        keyStr = SearchKey.value
        if (keyStr == "") {
            SearchList.innerHTML = ""
            return
        }
        if (event.keyCode !== 13) return
        SearchList.innerHTML = ""
        var elList = document.querySelectorAll('g');
        for (i = 0; i < elList.length; i++) {
            g = elList[i]
            if (g.classList.contains("graph")) continue;
            if (!(g.classList.contains("node"))) continue;
            a = g.querySelector("a")
            if (!a) continue
            title = g.querySelector("title")
            if (!title) continue
            tooltip_txt = a.getAttribute("xlink:title")
            if (!tooltip_txt) continue
            indexof = tooltip_txt.toLowerCase().indexOf(keyStr.toLowerCase())
            if (indexof >= 0) {
                li = document.createElement("li")
                li.innerHTML = title.innerHTML
                let target = { element: g };
                li.setAttribute("class", "search_li");
                var rect = g.getBoundingClientRect()
                li.setAttribute("targetRect_x", rect.left +  window.scrollX)
                li.setAttribute("targetRect_y", rect.top +  window.scrollY)
                li.onclick = function() {
                    var x = this.getAttribute("targetRect_x")
                    var y = this.getAttribute("targetRect_y")
                    window.scrollTo(x - window.innerWidth/2, y - window.innerHeight/2)
                    SearchPointer.style.top = y
                    SearchPointer.style.left = x
                }
                SearchList.appendChild(li)
            }
        }
        return;
    }
    let infobar_on = null

    infobar.addEventListener("click", function (event) {
        event.stopPropagation();
    })

    // make svg dragable
    svg = document.body.querySelector("svg")

    let pos = { top: 0, left: 0, x: 0, y: 0 };

    const mouseMoveHandler = function (e) {
        const dx = e.clientX - pos.x;
        const dy = e.clientY - pos.y;
        window.scroll(pos.left - dx, pos.top - dy);
    };

    const mouseUpHandler = function (e) {
        svg.onpointermove = null;
        svg.onpointerup = null;
        svg.releasePointerCapture(e.pointerId);
        svg.style.cursor = '';
        svg.style.removeProperty('user-select');
    };

    const mouseDownHandler = function (e, ele) {
        if (e.pointerType != "mouse") return;
        pos = {
            // The current scroll
            left: window.scrollX,
            top: window.scrollY,
            // Get the current mouse position
            x: e.clientX,
            y: e.clientY,
        };
        // Change the cursor and prevent user from selecting the text
        svg.style.cursor = 'grabbing';
        svg.style.userSelect = 'none';
        svg.setPointerCapture(e.pointerId);
        svg.onpointermove = mouseMoveHandler;
        svg.onpointerup = mouseUpHandler;
    };

    svg.classList.add("grabbable");
    svg.onpointerdown = mouseDownHandler;

    var elList = document.querySelectorAll('g');
    elList.forEach(
        function (g) {
            if (g.classList.contains("graph")) return;
            a = g.querySelector("a")
            if (!a) return
            tooltip_txt = a.getAttribute("xlink:title")
            if (!tooltip_txt) return
            g.style.cursor = "pointer"

            // prevent `pointerdown` being processed by `mouseDownHandler`
            // which in turn capture the pointer and fails to trigger g.onclick()
            g.onpointerdown = function (event) { event.stopPropagation(); }

            // pop up info bar
            g.onclick = function (event) {
                // toggle
                if (infobar_on === this) {
                    infobar_on = null
                    infobar.style.display = "none";
                } else {
                    infobar_on = this;
                    a = this.querySelector("a")
                    tooltip_txt = a.getAttribute("xlink:title")
                    var rect = this.getBoundingClientRect()
                    infopre.innerHTML = tooltip_txt
                    infobar.style.top = rect.top + window.scrollY
                    infobar.style.left = rect.right + window.scrollX
                    // remove custom setting, recover display's orginal setting in CSS
                    infobar.style.display = '';
                }
                
                if (event)
                    event.stopPropagation();
            }
        }
    )
</script>

</body>
</html>
'''
def visualize_model(model, fontsize=12, filename=None, detailed_label=False):
    g, data_map = generate_graph(model, fontsize, detailed_label=detailed_label)
    graph_src = Source(g.source, format="svg")
    if filename:
        svg = graph_src.pipe().decode('utf-8')
        if filename.endswith(".html"):
            m = re.search('(<svg(.|\n)*<\/svg>)', svg)
            assert(m)
            output_src = html_prefix + m.group(1) + html_surfix
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



def test_visualize():
    from openvino.runtime import Core, Model, Tensor, PartialShape, Type, Shape
    from openvino.runtime import opset8 as opset
    from openvino.runtime.passes import Manager
    import numpy as np, os, sys

    device = "CPU"
    core = Core()

    def simplecnn(weight_value, bias_value):
        input_shape = [1, 3, 16, 16]
        param_node = opset.parameter(dtype = np.float32, shape=input_shape)

        input = opset.fake_quantize(param_node,[0.],[255.],[0.],[255.], 256)
        c_out = 1
        c_in = input_shape[1]
        padding_begin = padding_end = [0, 0]
        wvalue = opset.constant(np.ones([c_out,c_in,1,1], dtype=np.int8)*weight_value)
        weight = opset.convert(wvalue, np.float32) - opset.constant(np.zeros([1,1,1,1], dtype=np.float32))
        weight = weight * opset.constant(np.ones([1,1,1,1], dtype=np.float32) * (1/weight_value))
        conv = opset.convolution(input, weight, [1, 1], padding_begin, padding_end, [1, 1], name="conv1")
        bias = opset.constant(np.ones([c_out,1,1], dtype=np.float32)*bias_value)
        add = opset.add(conv, bias)

        model = Model([add], [param_node], 'simplecnn')
        return model

    weight_value = 65
    bias_value = 0.123456
    model = simplecnn(weight_value, bias_value)
    os.environ["OPT_LINENUM"] = str(999)
    compiled_model_quant = core.compile_model(model, "CPU")

    input0 = np.zeros([1, 3, 16, 16], dtype = np.float32)
    expect0 = np.zeros([1, 1, 16, 16], dtype = np.float32)
    outputs = compiled_model_quant.infer_new_request({0: input0})
    out = outputs[compiled_model_quant.output(0)]
    gs, dt = compiled_model_quant.get_runtime_model().visualize()
    print(dt)
    sys.exit(0)


def test2():
    from openvino.runtime import Core, Model, Tensor, PartialShape, Type, Shape
    from openvino.runtime import opset8 as opset
    from openvino.runtime.passes import Manager
    import numpy as np
    import ovhelper, os

    device = "CPU"
    core = Core()

    def simplecnn():
        input_shape = [1, 32, 8]
        param1 = opset.parameter(dtype = np.float32, shape=input_shape)

        w0 = np.identity(8,dtype=np.float32); w0[1,1] = 2
        m0 = opset.matmul(param1, w0, False, False)

        w1 = np.identity(8,dtype=np.float32); w1[1,1] = 3
        m1 = opset.matmul(param1, w1, False, False)

        c0 = opset.constant(np.ones([1,1,1], dtype=np.float32) * (-1))
        c1 = opset.constant(np.ones([1,1,1], dtype=np.float32) * (0.123))

        result = (m0 * m1)+(m0*c0 + c1)*m1
        model = Model([result], [param1], 'simplecnn')
        return model

    model = simplecnn()
    #graph,data=model.visualize()

    os.environ["OPT_LINENUM"] = str(999)
    compiled_model_quant = core.compile_model(model, "CPU")
    compiled_model_quant.get_runtime_model().visualize(filename = "1.html")
    sys.exit(0)


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
    b = rs.uniform(rand_min, rand_max, list(shape)).astype(dtype)
    return ov.Tensor(a*alpha + b *(1-alpha))

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

    #test2()
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

    compiled_model = core.compile_model(model, "CPU")

    '''
    req = compiled_model.create_infer_request()
    for i in range(100):
        a = i*0.25 - 1
        all_input = {}
        for input in compiled_model.inputs:
            print("{:<10} {} {}".format(input.get_any_name(), input.get_element_type(), input.get_shape()))
            all_input[input] = fill_tensors_with_random(input, a)
        req.infer(inputs=all_input)
        print(a)
        from PIL import Image
        im = Image.fromarray(req.output_tensors[0].data.squeeze().astype(np.uint8))
        im.save("your_file_{}.png".format(a))
        if (a > 1.6):
            sys.exit(0) 
    '''
    latency_list, prof_list, fps, wtime = test_infer_queue(compiled_model, 2, 20000, time_limit=10)
    print(f"test_infer_queue FPS:{fps:.1f}")

    dest_file = filename="{}_{}.html".format(model_path, OPT_LINENUM)
    compiled_model.get_runtime_model().visualize(filename=dest_file)
    print("{} is saved!".format(dest_file))
