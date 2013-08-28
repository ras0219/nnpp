#include <fstream>
#include <vector>
#include <armadillo>
#include <random>
#include <iostream>
#include <iomanip>
#include <cassert>
#include <map>
#include <set>
#include <nnpp>

using namespace std;
using namespace arma;
using namespace nnpp;

typedef multilayer<sigmoid, sigmoid> xor_network;
typedef multilayer<fanout, xor_network> network;

std::default_random_engine g;
std::uniform_int_distribution<size_t> sz(1, 6);
std::uniform_int_distribution<size_t> d('a','z');

vector<size_t> random_word() {
  vector<size_t> r(6);
  size_t l = sz(g);
  for (size_t i = 0; i < l; ++i)
    r[i] = d(g);
  for (; l < 6; ++l)
    r[l] = '`';
  return r;
}

int main(int argc, char** argv)
{
  fstream simps("data/simpsons-wordlist.txt");

  vector< pair<string, size_t> > dict;

  string line;
  while (getline(simps, line)) {
    if (line.empty() or line[0] == '#')
      continue;
    if (line.back() == '\r')
      line.pop_back();
    auto f = line.find(' '); // find the second one
    if (f == string::npos)
      continue;
    auto s = line.find(' ', f + 1);
    if (s == string::npos)
      continue;
    dict.push_back(
      {   string(line.begin() + f + 1, line.begin() + s),
          stoi(string(line.begin() + s + 1, line.end()))});
  }

  cout << "Success reading in dictionary. " << dict.size() << " records." << endl;

  vector< pair< vector<size_t>, rowvec > > training_data;

  for (auto sp : dict) {
    vector<size_t> first;
    for (size_t i = 0; i < 6; ++i)
      if (i < sp.first.size())
        first.push_back(sp.first[i]);
      else
        first.push_back('`');
    training_data.push_back({first, { (double)1.0 }});
  }

  const size_t num_features = 3;

  network L(fanout(128, 6, num_features), xor_network(num_features * 6, 6, 1));

  const int max_x = 1001;

  for (int x = 0; x < max_x; ++x) {
    for (auto p : training_data) {
      L.train(p.first, p.second);
      L.train(random_word(), { 0.0 });
      L.train(random_word(), { 0.0 });
      L.train(random_word(), { 0.0 });
      L.train(random_word(), { 0.0 });
      L.train(random_word(), { 0.0 });
      L.train(random_word(), { 0.0 });
    }
    if (x % (max_x / 100) == 0)
      (cout << "\r => " << (x * 100 / max_x) << "% complete").flush();
  }
  cout << endl;

  cout << "Trained." << endl;

  mat m;

  double accept = 0.0;
  for (auto p : training_data) {
    typename network::output_type out = L.eval(p.first);
    typename network::output_type err = abs(round(out) - p.second);

    if (L.eval(p.first)[0] > 0.5) accept++;
//    cout << string(p.first.begin(), p.first.end()) << " - " << L.eval(p.first)[0] << endl;
  }
  accept /= training_data.size();
  cout << "Accepted: " << accept << endl;

  cout << "hidden layer weights" << endl;
  cout << L.l2.l1.m << endl;
  cout << "outer layer weights" << endl;
  cout << L.l2.l2.m << endl;

  set<char> classes[1 << num_features];
  multimap<double, char> feature[num_features];

  for (char c = '`'; c <= 'z'; ++c) {
    rowvec features = L.l1.t.eval(c);
    for (size_t i = 0; i < num_features; ++i)
      feature[i].insert({features[i], c});

    cout << c << fixed << setprecision(4);
    for (size_t i = 0; i < num_features; ++i)
      cout << " - " << setw(8) << features[i];
    cout << endl;

    int classnum = 0;
    for (size_t i = 0; i < num_features; ++i)
      classnum = (classnum * 2) + (int)round(features[i]);

    classes[classnum].insert(c);
  }


  bool sep;
  for (size_t i = 0; i < num_features; ++i) {
    sep = false;
    cout << "=== Feature " << i << endl;
    for (auto p : feature[i]) {
      if (!sep && p.first > 0.5) {
        cout << "------------" << endl;
        sep = true;
      }
      cout << p.second << " - " << setw(8) << p.first << endl;
    }
  }

  for (size_t i = 0; i < (1 << num_features); ++i) {
    cout << "=== Class " << hex << i << dec << endl;
    for (auto p : classes[i])
      cout << p << " ";
    cout << endl;
  }

  return 0;
}
