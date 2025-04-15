import metis

# 5 nodes in a circle
graph = metis.adjlist_to_metis(
    [[[1], [5]], [[0], [2, 2]], [[1, 2], [3]], [[2], [4]], [[3], [1]]]
)

part = metis.part_graph(graph, 5)
print(part)
