import faiss
import numpy as np
import time
import matplotlib.pyplot as plt

def benchmark_index(index, data, query, k=5):
    # Add vectors to the index
    start_time = time.time()
    index.add(data)
    index_time = time.time() - start_time

    # Perform search
    start_time = time.time()
    distances, indices = index.search(query, k)
    search_time = time.time() - start_time

    return index_time, search_time

def main():
    dimension = 128
    num_vectors = 10000
    num_queries = 100
    k = 5

    # Generate random data
    data = np.random.random((num_vectors, dimension)).astype('float32')
    query = np.random.random((num_queries, dimension)).astype('float32')

    # Index types to test
    index_types = [
        ('FlatL2', faiss.IndexFlatL2(dimension)),
        ('IVFFlat', faiss.IndexIVFFlat(faiss.IndexFlatL2(dimension), dimension, 100)),
        # ('IVFPQ', faiss.IndexIVFPQ(faiss.IndexFlatL2(dimension), dimension, 100, 8)),
        ('HNSW', faiss.IndexHNSWFlat(dimension, 32))
    ]

    indexing_times = []
    search_times = []
    labels = []

    for name, index in index_types:
        print(f"Benchmarking {name} index")

        # If using IVF-based indices, they need to be trained
        if 'IVF' in name:
            print("Training index...")
            index.train(data)

        index_time, search_time = benchmark_index(index, data, query, k)
        
        indexing_times.append(index_time)
        search_times.append(search_time)
        labels.append(name)
        
        print(f"Indexing time: {index_time:.4f} seconds")
        print(f"Search time: {search_time:.4f} seconds")
        print("-" * 40)

    # Plot results
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots()
    bars1 = ax.bar(x - width/2, indexing_times, width, label='Indexing Time')
    bars2 = ax.bar(x + width/2, search_times, width, label='Search Time')

    ax.set_xlabel('Index Type')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('FAISS Index Benchmarking')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    fig.tight_layout()

    plt.show()

if __name__ == "__main__":
    main()
