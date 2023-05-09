import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph()

G.add_node(1)
G.add_node(1)
G.add_node(2)
G.add_node(3)
G.add_node('A')
G.add_node('B')



G.add_edge(2, 1, weight=2)
if G.has_edge(1, 2):
    G[2][1]['weight'] += 1
else:
    G.add_edge(1, 2, weight=4)
G.add_edge(1, 9)

G.add_edge('A', 1)
G.add_edge('A', 'B')
G.add_edge(3, 'B')


print(list(G.nodes))
print(list(G.edges(data=True)))
nx.draw(G)
plt.savefig("path.png")

centrality = nx.eigenvector_centrality(G, weight='weight')

print(centrality[1])
print('\n')
for v, c in centrality.items():
    print(v, c)