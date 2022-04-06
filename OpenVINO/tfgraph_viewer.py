import pydot
from itertools import chain
import tensorflow as tf
import numpy as np

def visualize(filepath = None, all_ops = None, multilocation_ops = []):
    dot = pydot.Dot()
    dot.set('rankdir', 'UD')
    dot.set('concentrate', True)
    dot.set_node_defaults(shape='record')
    if all_ops is None:
        all_ops = tf.get_default_graph().get_operations()

    # eval all const tensor in "CPU" session
    with tf.Session() as sess:
        const_values = {}
        const_tensors = [c_op.outputs[0] for c_op in all_ops if c_op.type == "Const"]
        values = sess.run(const_tensors)
        for t, v in zip(const_tensors, values):
            const_values[t.name] = v

    # variable is not an OP but OP may read from it
    for var in tf.global_variables():
        node = pydot.Node(name = var.name,
                          label="variable",
                          tooltip= var.name,
                          margin = '"0,0"', width = "0", height = "0", style = '"filled"',
                          fillcolor = "brown1")
        dot.add_node(node)

    all_tensors = [k for k in set(chain(
                        *[c_op.outputs for c_op in all_ops],
                        *[c_op.inputs for c_op in all_ops],
                        ))]
    simple_const_op = []
    for c_node in all_tensors:
        # variable/read is not in all_ops
        if c_node.op not in all_ops: all_ops.append(c_node.op)
        label = c_node.op.type
        tooltip = "{}".format(c_node.op)
        tooltip = tooltip.replace('"',"'")
        color = "cyan"
        if c_node.name in const_values:
            val = const_values[c_node.name]
            cnt = np.prod(val.shape)
            is_int = np.issubdtype(val.dtype, np.integer)
            if (cnt <= 8 and is_int) or (cnt == 1):
                if is_int:
                    label = '"{}"'.format(",".join(["{}".format(v) for v in val.flatten()]))
                else:
                    label = '"{}"'.format(",".join(["{:.2f}".format(v) for v in val.flatten()]))
                    tooltip = np.array_repr(val, 100)
                color = "gray88"
                simple_const_op.append(c_node.op.name)
            else:
                vstr = np.array_repr(val, 100, 3).split('\n')
                if len(vstr) > 32:
                    vstr = [*vstr[:30],"...\n",vstr[-1]]
                tooltip = "\n".join(vstr)
        
        shape = "box"
        if c_node.op.name in multilocation_ops:
            shape = "cds"
            color = "cyan3"
        
        node = pydot.Node(name = c_node.op.name,
                            label=label,
                            tooltip=tooltip,
                            margin = '"0,0"', width = "0", height = "0", style = '"filled,rounded"',shape=shape,
                            fillcolor = color)
        dot.add_node(node)

    multilocation_id = [0 for _ in multilocation_ops]
    for c_op in all_ops:
        for io, c_output in enumerate(c_op.outputs):
            for ii, c_input in enumerate(c_op.inputs):
                label = []
                if c_input.op.name in simple_const_op:
                    pass
                else:
                    if len(c_input.shape) > 0:
                        label.append("{}".format(c_input.shape))
                    if c_input.dtype.name != "float32":
                        label.append("{}".format(c_input.dtype.name))
                src_name = c_input.op.name
                
                if src_name in multilocation_ops:
                    k = multilocation_ops.index(src_name)
                    src_id = multilocation_id[k]
                    multilocation_id[k] += 1
                    src_name = "{}[{}]".format(src_name, src_id)
                    node = pydot.Node(name = src_name,
                            label=c_input.op.type,
                            tooltip=tooltip,
                            margin = '"0,0"', width = "0", height = "0", style = '"filled,rounded"',
                            shape = "cds",
                            fillcolor = color)
                    dot.add_node(node)
                
                dot.add_edge(pydot.Edge(
                    src_name,
                    c_output.op.name,
                    label="\n".join(label),
                    fontsize="10",
                    labeltooltip=""))
    dot_src = dot.create("dot", "svg", encoding="utf-8")
    if isinstance(dot_src, bytes):
        dot_src = dot_src.decode('utf-8')
    import dot_svg_html
    dot_svg_html.dot_to_html(dot_src, filepath)
    return dot
