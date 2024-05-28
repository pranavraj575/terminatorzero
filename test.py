def all_DAG_subgraphs_with_property(edge_list: dict, used_sources=None):
    """
    iterable of all lists of edges that create a DAG such that each vertex is the source of at most one edge
        (we run through all permutations, sort of wasteful)
    the order returned will be in reverse topological order

    :param edge_list: dict(vertex -> vertex set), must be copyable
    :return: iterable of (list[(start vertex, end vertex)], used vertices)
    """
    if used_sources is None:
        used_sources = set()
    yield (),used_sources
    for source in edge_list:
        if source not in used_sources:
            for end in edge_list[source]:
                for (subsub, now_used) in all_DAG_subgraphs_with_property(edge_list=edge_list,
                                                                          used_sources=
                                                                          used_sources.union({source, end})
                                                                          ):
                    yield (((source, end),)+subsub ,
                           now_used)
edge_list={
    0:{1},
    1:{2},
    2:{3,},
    3:{0,4},
    4:{5},
}
for thing in all_DAG_subgraphs_with_property(edge_list):
    print(thing)

print(len(list(all_DAG_subgraphs_with_property(edge_list))))