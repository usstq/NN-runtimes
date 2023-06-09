Writing a openVINO transformation is not easy task, especially when the pattern is very complex, developer needs to closely check the network using netron for the pattern to be matched, and translate them into a serials of ngraph API calls, it's time-consuming and error-prone.

## Proposal 1
But the work can be greatly simplified if we only specify a sub-set of nodes to be matched, and their attributes and connections/relation-ships can be extracted from the orginal network topology instead.

This can be even simpler if it can be done interactively, for example, we check the network using netron, and copy the nodes to be collected, and generate the matcher automatically and run, and then a rewritten network can be generted and rechecked in netron to see if our pattern is replaced indeed. and we can repeat this procedure until the interested subgraph is all included in the pattern.

To accomplish this, we need:
 - visualization tools: netron
 - serialization tools: ov::serialize
 - generation tools :
    - given node names and orginal model IR (xml/bin), extract minimal sub-set of nodes which contains the given nodes.
    - serialize the sub-nodes & connections into human readible format
    - in C++, build the subgraph from the human readible format string

So develop a generation tool is required, it can read-in XML IR and find the sub-set of nodes and save them into another IR + a format that can be deserialized into the pattern again.

We need an [SSA](https://en.wikipedia.org/wiki/Static_single-assignment_form) style IR that:
 - is human read-able & editable
 - contains simple constant number
 - contains partial shapes which only for human read purpose, when deserialized back to ngraph structure, only meaningfull for input node(w/o producer)
 - contains serizalied attributes (number,string,enum...) because it's part of the semantic definition of OPs
 - is serializable to/from ngraph data structure and thus can be saved to IR/XML and visualized using netron



## Proposal 2

Study IPEX's code: https://github.com/intel/intel-extension-for-pytorch/blob/9608313cb2466c591f83a5f02604ccefced0f958/csrc/jit/passes/graph_rewrite_mha.cpp#L42 we can see it seems to be more clear from the code, thus the maintainence is better, so another solution would be conquer complex pattern stage by stage, for example, one-pass can greatly simplify the ShapeOf-subgraph, purely for optimizing the semantics and make following pattern simpler. since most ShapeOf subgraph can be combined into a ShapeInfer node, which took shapes of some inputs and derive output shape using some rules defined by a string:

for example: `[i0.shape[0], 1, 1, i1.shape[1]]` can represent shape of 

