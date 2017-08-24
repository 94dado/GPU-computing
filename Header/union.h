#ifndef _union_
#define _union_ 1

#include <iostream>
#include <vector>

// A Union-Find structure is needed by Kruskal's algorithm.
// You can google it to understand how it works.
// Basicly, it efficiently manages disjoint sets, making it possible to
// join them and check if two elements belong to the same set.
// find(A) = find(B) IFF A and B belong to the same set.
class Union_Find {
    std::vector<int> _sets;
    int _number_of_sets;

    public:
      Union_Find() {
        _number_of_sets = 0;
      }

      void Reset (int number_of_sets) {
        _sets.clear();
        _sets.resize(number_of_sets, -1);
        _number_of_sets = number_of_sets;
      }

      bool Union(int set_a, int set_b) {
        int root_a = Find(set_a), root_b = Find(set_b);
        if (root_a == root_b) return false;
        _sets[root_a] += _sets[root_b];
        _sets[root_b] = root_a;
        _number_of_sets--;
        return true;
      }

      int Find(int element) {
        if (_sets[element] < 0) return element;
        return (_sets[element] = Find(_sets[element]));
      }

      inline int GetNumberOfSets() {
        return _number_of_sets;
      }
};

#endif
