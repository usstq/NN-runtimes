# import openvino.runtime as ov
import numpy as np
from graphviz import Digraph, Source

# print Model in readable text
def print_model(model):
    out2name = {}
    nameid = 1

    ilist = [i.get_node().get_name() for i in model.inputs]
    print("model({}):".format(",".join(ilist)))
    for n in model.get_ordered_ops():
        # collect output and also allocate output names
        returns = []
        for out in n.outputs():
            
            varname = "t{}".format(nameid)
            returns.append("Tensor<{}x{}> {}".format(
                                "x".join([str(s) for s in out.get_shape()]),
                                out.get_element_type().get_type_name(),
                                varname))
            out2name[out] = varname
            nameid += 1
        # collect source output names of corresponding inputs
        args = [out2name[i.get_source_output()] for i in n.inputs()]

        # generate psuedo code
        attrs = ["{}={}".format(k, v) for k,v in n.get_attributes().items()]
        rt_info = ["{}={}".format(k, v) for k,v in n.get_rt_info().items()]
        opname = n.get_type_name()
        friendly_name = n.get_friendly_name()
        if opname == "ExecutionNode" or friendly_name.startswith(opname):
            opname = friendly_name
            friendly_name = ""

        print("    {} = {}({}{}) {}".format(
                        ",".join(returns),
                        opname,
                        ",".join(args),
                        "" if len(attrs) == 0 else ("," if len(args)>0 else "") + (",".join(attrs)),
                        "" if len(friendly_name)==0 else "   # {}".format(friendly_name) ))
        if (rt_info):
            print("\t\t\t#rt_info:\n\t\t\t#\t{}\n".format("\n\t\t\t#\t".join(rt_info)))

    olist = [out2name[i] for i in model.outputs]
    print("    return {}".format(",".join(olist)))

def visualize_model(model, fontsize=12, filename=None):
    # create all nodes before edges
    g = Digraph("openvino models")
    node2name = {}
    nodenames = set()
    op2color = {"Parameter":"gold", "Result":"deeppink", "Constant":"gray"}
    for n in model.get_ordered_ops():
        opname = n.get_type_name()
        friendly_name = n.get_friendly_name()
        rt_info = n.get_rt_info()
        attrs = ["{}={}".format(k, v) for k,v in n.get_attributes().items()]
        rtinfo = ["{}={}".format(k, v) for k,v in rt_info.items()]

        # ExecutionNode is fake wrapper of runtime node
        # and this type name gives less information than friendly_name
        if opname == "ExecutionNode":
            opname = friendly_name
        
        # rt_info["layerType"] gives OP type information
        if "layerType" in rt_info:
            opname = rt_info["layerType"]

        label = opname

        # originalLayersNames gives mapping between runtime nodes and orginal nodes
        if "originalLayersNames" in rt_info:
            label = "{"+opname+"|"+rt_info['originalLayersNames'].replace(",","|") +"}"

        allinfo = friendly_name
        if (attrs):
            allinfo += "\n----attributes----\n{}".format("\n".join(attrs))
        if (rt_info):
            allinfo += "\n----rt_info----\n{}".format("\n".join(rtinfo))

        color = op2color[opname] if opname in op2color else "cyan"
        g.node(name=friendly_name,
                label=label,
               shape='Mrecord',
               style='filled,rounded',
               fillcolor=color,
               fontsize=str(fontsize),
               margin="0,0",width="0",height="0",
               tooltip=allinfo)
        assert(friendly_name not in nodenames) # make sure the name is uinque
        nodenames.add(friendly_name)
        node2name[n] = friendly_name
    for n in model.get_ordered_ops():
        for i in n.inputs():
            str_shape = ",".join([str(s) for s in i.get_shape()])
            str_ele_type = i.get_element_type().get_type_name()
            src_rt_info = i.get_source_output().get_node().get_rt_info()

            label = '[{}]'.format(str_shape)
            if "outputLayouts" in src_rt_info:
                label += " " + src_rt_info["outputLayouts"]
            if "outputPrecisions" in src_rt_info:
                label += " " + src_rt_info["outputPrecisions"]
            else:
                label += " " + str_ele_type
            g.edge(
                i.get_source_output().get_node().get_friendly_name(),
                n.get_friendly_name(),
                label=label,
                color='black',
                fontsize=str(fontsize))
    graph_src = Source(g, format="svg")
    if filename:
        return graph_src.render(filename)
    return graph_src

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

if __name__ == "__main__":
    pass