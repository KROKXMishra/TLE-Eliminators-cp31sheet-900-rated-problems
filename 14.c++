// ============================================================================
// COMPETITIVE PROGRAMMING TEMPLATES & UTILITIES
// ============================================================================

#include <bits/stdc++.h>
using namespace std;

typedef long long ll;
typedef unsigned long long ull;
typedef long double ld;
typedef pair<int, int> pii;
typedef pair<ll, ll> pll;
typedef vector<int> vi;
typedef vector<ll> vll;
typedef vector<string> vs;
typedef vector<char> vc;

#define pb push_back
#define mp make_pair
#define fi first
#define se second
#define all(v) v.begin(), v.end()
#define rall(v) v.rbegin(), v.rend()
#define sz(v) (int)v.size()
#define clr(v, x) memset(v, x, sizeof(v))
#define rep(i, a, b) for(int i = a; i < b; i++)
#define rrep(i, a, b) for(int i = a; i >= b; i--)

const ll MOD = 1e9 + 7;
const ll INF = 1e18;
const int MAXN = 2e5 + 5;

// ============================================================================
// 1. MATH UTILITIES
// ============================================================================

// GCD using Euclidean algorithm
ll gcd(ll a, ll b) {
    return b == 0 ? a : gcd(b, a % b);
}

// LCM
ll lcm(ll a, ll b) {
    return a / gcd(a, b) * b;
}

// Power with modulo
ll power(ll a, ll b, ll mod) {
    ll res = 1;
    a %= mod;
    while (b > 0) {
        if (b & 1) res = (res * a) % mod;
        a = (a * a) % mod;
        b >>= 1;
    }
    return res;
}

// Modular inverse (using Fermat's little theorem, requires MOD to be prime)
ll modInverse(ll a, ll mod) {
    return power(a, mod - 2, mod);
}

// Factorial and inverse factorial precomputation
vector<ll> fact(MAXN), inv_fact(MAXN);

void precompute_factorials() {
    fact[0] = 1;
    for (int i = 1; i < MAXN; i++) {
        fact[i] = (fact[i-1] * i) % MOD;
    }
    inv_fact[MAXN-1] = modInverse(fact[MAXN-1], MOD);
    for (int i = MAXN-2; i >= 0; i--) {
        inv_fact[i] = (inv_fact[i+1] * (i+1)) % MOD;
    }
}

// nCr (n choose r) with modulo
ll nCr(int n, int r) {
    if (r > n || r < 0) return 0;
    return (fact[n] * inv_fact[r] % MOD) * inv_fact[n-r] % MOD;
}

// Check if number is prime
bool isPrime(ll n) {
    if (n < 2) return false;
    if (n == 2) return true;
    if (n % 2 == 0) return false;
    for (ll i = 3; i * i <= n; i += 2) {
        if (n % i == 0) return false;
    }
    return true;
}

// Get all divisors
vi getDivisors(int n) {
    vi divisors;
    for (int i = 1; i * i <= n; i++) {
        if (n % i == 0) {
            divisors.pb(i);
            if (i != n / i) divisors.pb(n / i);
        }
    }
    sort(all(divisors));
    return divisors;
}

// ============================================================================
// 2. ARRAY/STRING UTILITIES
// ============================================================================

// Binary search (find first occurrence >= x)
int binarySearch(vi &arr, int x) {
    int lo = 0, hi = sz(arr) - 1, ans = -1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (arr[mid] >= x) {
            ans = mid;
            hi = mid - 1;
        } else {
            lo = mid + 1;
        }
    }
    return ans;
}

// Merge two sorted arrays
vi mergeSortedArrays(vi &a, vi &b) {
    vi result;
    int i = 0, j = 0;
    while (i < sz(a) && j < sz(b)) {
        if (a[i] <= b[j]) result.pb(a[i++]);
        else result.pb(b[j++]);
    }
    while (i < sz(a)) result.pb(a[i++]);
    while (j < sz(b)) result.pb(b[j++]);
    return result;
}

// Check if string is palindrome
bool isPalindrome(string s) {
    int n = sz(s);
    for (int i = 0; i < n / 2; i++) {
        if (s[i] != s[n-1-i]) return false;
    }
    return true;
}

// Convert string to integer
ll stringToInt(string s) {
    ll num = 0;
    for (char c : s) {
        num = num * 10 + (c - '0');
    }
    return num;
}

bool kmp(string& x,string& s,int n,int m){
    vi lps(m,0);
    
    int i=0,j=1;
    while(j<m){
        if(s[i]==s[j]){
            lps[j]=i+1;
            i++;j++;
        }
        else{
            if(i==0){
                lps[j]=0;
                j++;
            }
            else{
                i=lps[i-1];
            }
        }
    }

    i=0;j=0;
    while(i<n && j<m){
        if(x[i]==s[j]){
            i++;
            j++;
        }
        else{
            if(j==0){
                i++;
            }
            else{
                j=lps[j-1];
            }
        }
    }
    return(j==m);
}

// ============================================================================
// 3. PREFIX SUM / DIFFERENCE ARRAY
// ============================================================================

// Prefix sum array
vi prefixSum(vi &arr) {
    vi prefix(sz(arr) + 1, 0);
    for (int i = 0; i < sz(arr); i++) {
        prefix[i+1] = prefix[i] + arr[i];
    }
    return prefix;
}

// Range sum query using prefix sum
ll rangeSum(vi &prefix, int l, int r) {
    return prefix[r+1] - prefix[l];
}

// Difference array for range updates
void updateRange(vi &diff, int l, int r, int val) {
    diff[l] += val;
    diff[r+1] -= val;
}

// Convert difference array back to normal array
vi getDifferenceArray(vi &diff) {
    vi result;
    int curr = 0;
    for (int d : diff) {
        curr += d;
        result.pb(curr);
    }
    return result;
}

// ============================================================================
// 4. GRAPH UTILITIES (Adjacency List)
// ============================================================================

vector<int> adj[MAXN];
bool visited[MAXN];
int dist[MAXN];
int parent[MAXN];

// DFS - Depth First Search
void dfs(int u) {
    visited[u] = true;
    for (int v : adj[u]) {
        if (!visited[v]) {
            dfs(v);
        }
    }
}

// BFS - Breadth First Search
void bfs(int start) {
    queue<int> q;
    q.push(start);
    visited[start] = true;
    dist[start] = 0;
    
    while (!q.empty()) {
        int u = q.front();
        q.pop();
        
        for (int v : adj[u]) {
            if (!visited[v]) {
                visited[v] = true;
                dist[v] = dist[u] + 1;
                parent[v] = u;
                q.push(v);
            }
        }
    }
}

// Topological Sort (using DFS)
void topoSort(int u, vector<bool> &vis, stack<int> &st) {
    vis[u] = true;
    for (int v : adj[u]) {
        if (!vis[v]) topoSort(v, vis, st);
    }
    st.push(u);
}

// ============================================================================
// 5. TWO POINTER TECHNIQUE
// ============================================================================

// Two pointer - Find pair with given sum
pair<int, int> findPairWithSum(vi &arr, int target) {
    int n = sz(arr);
    int left = 0, right = n - 1;
    
    while (left < right) {
        int sum = arr[left] + arr[right];
        if (sum == target) return {arr[left], arr[right]};
        else if (sum < target) left++;
        else right--;
    }
    return {-1, -1};
}

// ============================================================================
// 6. SORTING & COMPARATORS
// ============================================================================

// Comparator for sorting pairs by first element (ascending)
bool cmpPair(pii a, pii b) {
    return a.fi < b.fi;
}

// Comparator for custom structure
struct Person {
    string name;
    int age;
    bool operator<(const Person &other) const {
        return age < other.age;
    }
};

// ============================================================================
// 7. SEGMENT TREE
// ============================================================================

class SegmentTree {
private:
    vector<ll> tree;
    int n;
    
    void build(vector<ll> &arr, int node, int start, int end) {
        if (start == end) {
            tree[node] = arr[start];
        } else {
            int mid = (start + end) / 2;
            build(arr, 2*node, start, mid);
            build(arr, 2*node+1, mid+1, end);
            tree[node] = tree[2*node] + tree[2*node+1];
        }
    }
    
    void update(int node, int start, int end, int idx, ll val) {
        if (start == end) {
            tree[node] = val;
        } else {
            int mid = (start + end) / 2;
            if (idx <= mid) {
                update(2*node, start, mid, idx, val);
            } else {
                update(2*node+1, mid+1, end, idx, val);
            }
            tree[node] = tree[2*node] + tree[2*node+1];
        }
    }
    
    ll query(int node, int start, int end, int l, int r) {
        if (r < start || end < l) return 0;
        if (l <= start && end <= r) return tree[node];
        int mid = (start + end) / 2;
        return query(2*node, start, mid, l, r) + 
               query(2*node+1, mid+1, end, l, r);
    }

public:
    SegmentTree(vector<ll> &arr) {
        n = sz(arr);
        tree.resize(4 * n);
        build(arr, 1, 0, n-1);
    }
    
    void update(int idx, ll val) {
        update(1, 0, n-1, idx, val);
    }
    
    ll query(int l, int r) {
        return query(1, 0, n-1, l, r);
    }
};

// ============================================================================
// 8. FENWICK TREE (Binary Indexed Tree)
// ============================================================================

class FenwickTree {
private:
    vector<ll> tree;
    int n;
    
public:
    FenwickTree(int size) {
        n = size;
        tree.assign(n + 1, 0);
    }
    
    void update(int idx, ll delta) {
        for (int i = idx; i <= n; i += i & (-i)) {
            tree[i] += delta;
        }
    }
    
    ll query(int idx) {
        ll sum = 0;
        for (int i = idx; i > 0; i -= i & (-i)) {
            sum += tree[i];
        }
        return sum;
    }
    
    ll rangeQuery(int l, int r) {
        return query(r) - (l > 1 ? query(l - 1) : 0);
    }
};

// ============================================================================
// 9. UNION-FIND (DISJOINT SET UNION)
// ============================================================================

class DSU {
private:
    vector<int> parent, rank;
    
public:
    DSU(int n) {
        parent.resize(n);
        rank.assign(n, 0);
        for (int i = 0; i < n; i++) {
            parent[i] = i;
        }
    }
    
    int find(int x) {
        if (parent[x] != x) {
            parent[x] = find(parent[x]);  // Path compression
        }
        return parent[x];
    }
    
    bool unite(int x, int y) {
        int px = find(x), py = find(y);
        if (px == py) return false;
        
        // Union by rank
        if (rank[px] < rank[py]) swap(px, py);
        parent[py] = px;
        if (rank[px] == rank[py]) rank[px]++;
        return true;
    }
};

// ============================================================================
// 10. BASIC TEMPLATE
// ============================================================================


void solve() {
    ll n;
    cin >>n;
    ll a=0,b=0,k=n;
    if(n%2==1 || n<=3 || n==5){
        cout<<-1<<endl;
        return;
    }
    while(k>=0 && k%4!=0) {
        k-=6;
        b++;
    }
    ll maxi=(k/4)+b;
    k=n;
    while(k>=0 && k%6!=0) {
        k-=4;
        a++;
    }
    ll mini=(k/6)+a;
    cout<<mini<<" "<<maxi<<endl;
}
int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    
    int t;
    cin >> t;
    while (t--) {
        solve();
    }
    
    return 0;
}

// ============================================================================
// NOTES & TIPS
// ============================================================================
/*
 * 1. Always use "ios_base::sync_with_stdio(false); cin.tie(NULL);" for faster I/O
 * 
 * 2. Use long long (ll) for large numbers to avoid overflow
 * 
 * 3. Common complexities:
 *    - O(n log n): Sorting, Binary search with array
 *    - O(n): Linear scan
 *    - O(n^2): Nested loops
 *    - O(2^n): Brute force with bitmasks
 * 
 * 4. Space complexities to watch:
 *    - Vector of vectors: can use lots of memory
 *    - Adjacency list is generally better than matrix for sparse graphs
 * 
 * 5. Useful STL functions:
 *    - sort(all(v)): Sort array/vector
 *    - lower_bound(all(v), x): First element >= x
 *    - upper_bound(all(v), x): First element > x
 *    - binary_search(all(v), x): Check if x exists
 *    - max_element(all(v)): Find max
 *    - min_element(all(v)): Find min
 *    - accumulate(all(v), 0): Sum all elements
 * 
 * 6. Modulo tips:
 *    - Always take modulo at each step to avoid overflow
 *    - For subtraction: (a - b + MOD) % MOD
 *    - For multiplication: ((a % MOD) * (b % MOD)) % MOD
 */