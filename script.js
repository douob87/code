// Code data array
const codeData = [
    {
        title: "BFS 廣度優先搜索",
        category: "圖論",
        code: `//從(0,0) 走到 (n-1, m-1) 走幾步; 0代表路， 1代表牆
using T = array<int, 3>;
int n = mat.size(), m = mat[0].size();
vector<int> dir = {1, 0, -1, 0, 1};
queue<T> q;
q.push({0, 0, 0});

while(!q.empty()){
    auto [step, i, j] = q.front();q.pop();
    if(mat[i][j]==1)continue;
    mat[i][j] = 1;
    if(i==n-1 && j==m-1)return step; 
    for(int d=0;d<4;d++){
        int ni = i + dir[d];
        int nj = j + dir[d+1];
        if(ni>=0 && nj>=0 && ni<n && nj<m && !mat[ni][nj]){
            q.push({step+1, ni, nj});
        }
    }
}
return -1;`
    },
    {
        title: "DFS 時間戳",
        category: "圖論",
        code: `int t = 0//全局時間戳
int in[n], out[x];

function<void(int, int)> dfs = [&](int x, int fa){
    in[x] = ++t;
    for(int y : g[x]){
        if(y != fa){
            dfs(y, x);
        }
    }
    out[x] = t;
};
dfs(0, -1);

//判斷 x 是否為 y 的父節點
//1.in[x] <= in[y]; 父節點的進入時間 一定小於等於 子節點進入時間
//2.out[y] <= out[x]; 子節點離開時間 一定小於等於 父節點離開時間
//3.in[y] <= out[y]; 節點進入時間 一定小於等於 離開時間
//結論 => in[x] <= in[y] <= out[x]
function<bool(int, int)> isParent = [&](int x, int y)->bool{
    return in[x] <= in[y] && in[y] <= out[x];
};

//dfs() 的先後代表意義

//先遞迴 (後序 bottom - up)
//先求 子節點 相關的資訊
for(int y : g[x]){
    if(y != fa){
        dfs(y, x);
        //do something
    }
}

//後遞迴 (前序 top - down)
//把 父節點 的資訊往下傳
for(int y : g[x]){
    if(y != fa){
        //do something
        dfs(y, x);
    }
}`
    },
    {
        title: "Dijkstra 最短路徑",
        category: "圖論",
        code: `//找 *(0, 0) -> (n-1, m-1) 最大路徑總和
//適用 邊權值>=0 
//TC : O(|V+E|logE)
using T = array<int, 3>;
int n = grid.size(), m = grid[0].size();
vector<vector<int>> dist(n, vector<int>(m, INT_MAX));
vector<int> dir = {1, 0, -1, 0, 1};
priority_queue<T> pq;
pq.push({grid[0][0], 0, 0});
dist[0][0] = grid[0][0];

while(!q.empty()){
    auto [step, i, j] = pq.top();pq.pop();
    if(step < dist[i][j])continue;//當前走過更少步 沒必要再走
    if(i==n-1 && j==m-1)return step; 

    for(int d=0;d<4;d++){
        int ni = i + dir[d];
        int nj = j + dir[d+1];
        if(ni>=0 && nj>=0 && ni<n && nj<m){
            int new_step = step + grid[ni][nj];
            if(new_step > dist[ni][nj]){
                dist[ni][nj] = new_step;
                pq.push({new_step, ni, nj});
            }
        }
    }
}
return -1;`
    },
    {
        title: "LCA 最近公共祖先",
        category: "圖論",
        code: `/*
         8
        / \\
       5   7
      / \\ / \\
     1  2 3  4 
       / \\   
      8   6
                                  2
上圖 最深節點的最近公共祖先(LCA) = /  \\ 
                                8   6
思考點 : 若左子樹高度 == 右子樹高度 => LCA 為 {當前子樹}
             //      >     //    =>  LCA 為 {左子樹}
             //      <     //    =>  LCA 為 {右子樹}
*/
pair<TreeNode*, int> dfs(TreeNode* root){
    if(!root)return {nullptr, 0};
    auto [t1, d1] = dfs(root->left);
    auto [t2, d2] = dfs(root->right);
    if(d1 ==d2){//左子樹高度 == 右子樹高度
        return {root, d1+1};
    }else if(d1 > d2){//左子樹高度 > 右子樹高度
        return {left, d1+1};
    }else{//左子樹高度 < 右子樹高度
        return {right, d2+1};
    }
    //return {d1==d2 ? root : d1>d2 ? t1 : t2, max(d1, d2)+1};
}
/***********************************************************/
//倍增算法(Binary Lifting)
class TreeAncestor {
public:
    vector<vector<int>> fa;
    TreeAncestor(int n, vector<int>& parent) {
        int m = log2(n) + 1;
        fa.assign(n, vector<int>(m, -1));
        //base case
        for(int i=0;i<n;i++)fa[i][0] = parent[i];
        //fa[i][j] 表示 第 i 個node 往上 2^j 是誰？
        //fa[x][0] = parent[x]
        //fa[x][1] = fa[fa[x][0]][0]
        //轉移方程 : fa[x][i+1] = fa[fa[x][i]][i]
        for(int i=0;i<m-1;i++){
            for(int x=0;x<n;x++){
                int p = fa[x][i];
                if(p != -1){
                    fa[x][i+1] = fa[p][i];
                }
            }
        }
    }
    
    int getKthAncestor(int node, int k) {
        int m = log2(k) + 1;
        for(int i=0;i<m;i++){
            if(k>>i & 1){
                node = fa[node][i];
            }
            if(node < 0)break;
        }
        return node;
    }
};`
    },
    {
        title: "並查集 Union-Find",
        category: "圖論",
        code: `class UnionFind{
    vector<int> parent;
public:
    vector<int> conn;
    int cc;   
    UnionFind(int n){
        conn.assign(n, 0)
        parent.assign(n, 0);
        iota(parent.begin(), parent.end(), 0);
        cc = n;
    }
    void Union(int x, int y){
        x = Find(x);
        y = Find(y);
        if(x != y){
            parent[x] = y;
            conn[y] += conn[x];
            cc --;
        }
    }
    int Find(int x){
        return parent[x]==x ? x : parent[x]=Find(parent[x]);
    }
    void Reset(int x){
        parent[x] = x;
    }
    bool Connected(int x, int y){
        return Find(x) == Find(y);
    }
};
/*
UnionFind uf(n);
uf.cc; => 連通塊個數
uf.Union(u, v); => merge 兩個連通塊
uf.Find(node); => 找到node的祖先節點
uf.Reset(node); => 刪除邊
uf.Connected(u, v); => 確認u, v是否在同一個連通塊
uf.conn[Find(node)]; => node的連通塊個數
*/`
    },
    {
        title: "二分圖判定",
        category: "圖論",
        code: `//判斷graph(無向圖)是不是二分圖, 0(未塗色)，1(塗紅色)，-1(塗藍色)
int n;//vertex個數
vector<vector<int>> graph;
vector<int> color(n, 0);
bool dfs(int x, int c){
    color[x] = c;
    for(const int& y : graph[x]){
        if(color[y]==c || color[y]==0&&!dfs(y, -c))return false;
    }
    return true;
}
for(int i=0;i<n;i++){
    if(color[i]==0 && !dfs(i, 1))return false;
}
return true;`
    },
    {
        title: "Floyd-Warshall 全源最短路徑",
        category: "圖論",
        code: `//全源最短路徑
//枚舉中繼點 TC : O(n^3) || SC : O(n^2)
//想法 i->j = i->k->j
//轉移方程 => dp[i][j] = min(dp[i][j], dp[i][k] + dp[k][j]);
for(int k=0;k<n;k++){
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            dp[i][j] = min(dp[i][j], dp[i][k] + dp[k][j]);
        }
    }
}`
    },
    {
        title: "分組循環技巧",
        category: "小技巧",
        code: `//111000011100011
for(int i=0;i<n;){
    int j = i;
    for(;j<n && s[i]==s[j];j++);
    //cout<<s[i]<<" "<<"長度 : "<<j-i+1<<endl;
    i = j;
}`
    },
    {
        title: "埃氏篩法 Sieve of Eratosthenes",
        category: "數學",
        code: `int MX = 1'000'01;
vector<bool> sieve(MX, true);
sieve[0] = sieve[1] = false;

for(int i=2;i<MX;i++){
    if(!sieve[i])continue;
    for(long long j=1LL*i*i;j<MX;j+=i){//j 改 long long , 不然i*i會爆掉
        sieve[j] = false;
    }
}

/*
sieve[x] = true => 質數
sieve[x] = false => 非質數
*/`
    },
    {
        title: "平方剩餘核",
        category: "數學",
        code: `//把一個數 質因數分解，次方數是偶數=>次方變0 ; 次方數是奇數=>次方變1
//72 = 2^3 * 3^2 => 2
int MX = 1'000'01;
vector<int> core(MX, 0);

for(int i=2;i<MX;i++){
    if(core[i] != 0)continue;//已經被標記
    for(long long j=1;i*j*j<MX;j+=i){
        core[i*j*j] = i;
    }
}`
    },
    {
        title: "最小質數篩法",
        category: "數學",
        code: `int MX = 1'000'01;
vector<int> spf(MX, 0);
for(int i=2;i<MX;i++){
    //已經被標記
    if(spf[i] != 0)continue;
    for(long long j=i;j<MX;j+=i){
        //還沒被標記 就修改
        if(spf[j] == 0){
            spf[j] = i;
        }
    }
}
//質因數分解 => 20 = 2 * 2 * 5
int x = 2486;
while(x != 1){
    cout<<spf[x]<<endl;
    x /= spf[x];
}`
    },
    {
        title: "模逆元 Modular Inverse",
        category: "數學",
        code: `//(a/b) % p => 轉換成 a * b^-1(a 乘上 b的逆元)
//b = qpow(b, MOD-2) % MOD, 費馬小定理...
int MOD = 1'000'000'007;//1e9 + 7
int MX = 41;
int F[MX];
int INV_F[MX];

long long qpow(int a, int n){
    long long res = 1;
    while(n){
        if(n & 1){
            res = (res * a) % MOD;
        }
        a = (a * a) % MOD;
        n >>= 1;
    }
    return res;
}

F[0] = 1;
for(int i=1;i<MX;i++)F[i] = i * F[i-1] % MOD;
INV_F[MX-1] = qpow(F[MX-1], MOD-2);
for(int i=MX-1;i;i--)INV_F[i-1] = INV_F[i] * i % MOD;//計算模逆元，1 / (1*2) = 1 / (1*2*3) * 3

/*
C(n, m) = n! / (m)! / (n-m)!
計算C(n, m) % MOD = F[n] * INV_F[m] % MOD * INV_F[n-m] % MOD;
*/`
    },
    {
        title: "生成回文數",
        category: "數學",
        code: `const int MX = 5000, BASE = 10;
vector<int> pal;
void generate_pal(){
    pal = {0};//Dummy node
    for(int p=1; ;p*=BASE){
        //生成奇數回文
        for(int i=p;i<p*BASE;i++){
            int x = i;
            for(int t=i/BASE;t;t/=BASE){
                x = x*BASE + t%BASE;
            }
            if(x > MX){
                pal.push_back(5005);
                return ;
            }
            pal.push_back(x);
        }
        //生成偶數回文
        for(int i=p;i<p*BASE;i++){
            int x = i;
            for(int t=i;t;t/=BASE){
                x = x*BASE + t%BASE;
            }
            if(x > MX){
                pal.push_back(5005);
                return ;
            }
            pal.push_back(x);
        }
    }
}
/*
要找最近的回文數的話, 11, 22, 33, 44, 55, >= 31的value在22 33之間。
int j = lower_bound(pal.begin(), pal.end(), x) - pal.begin();
x 一定在 pal[j-1] ~ pal[j] 之間 
*/`
    },
    {
        title: "矩陣快速冪",
        category: "數學",
        code: `//計算 a ^ n , O(logn)
int qpow(int a, int n){
    int res = 1;
    while(n){
        if(n & 1){
            res *= a;
        }
        a *= a;
        n >>= 1;
    }
    return res;
}

//矩陣版
const int N = 2;
int MOD;
using Matrix = array<array<long long, N>, N>;
Matrix mul(Matrix& a, Matrix& b){
    Matrix c{};
    for(int i=0;i<N;i++){
        for(int k=0;k<N;k++){
            if(a[i][k] == 0)continue;
            for(int j=0;j<N;j++){
                c[i][j] = (c[i][j] + a[i][k] * b[k][j] % MOD) % MOD;
            }
        }
    }
    return c;
}
Matrix qpow(Matrix& a, int n){
    Matrix res{};
    for(int i=0;i<N;i++)res[i][i] = 1;
    while(n){
        if(n & 1){
            res = mul(res, a);
        }
        a = mul(a, a);
        n >>= 1;
    }
    return res;
}`
    },
    {
        title: "Log Trick 優化技巧",
        category: "演算法",
        code: `/*
TC : O(31n)

OR : 如果"新的num" 加到 一個OR sum的集合 不會改變值，代表接下來的OR sum都會 "被包含"。所以 (nums[i]|nums[j]) == nums[j] 要 break
AND : 如果"新的num" 加到 一個AND sum的集合 不會改變值，代表接下來的AND sum都會 "包含"。所以 ......
GCD : 同理。
LCM : 同理。
*/
for(int i=0;i<n;i++){
    for(int j=i-1;j>=0;j--){
        if((nums[i] | nums[j]) == nums[j])break;
        nums[j] = nums[i] | nums[j];
    }
}`
    },
    {
        title: "稀疏表 ST Table",
        category: "演算法",
        code: `//稀疏表 TC : O(n logn) || SC : O(n logn)
//查找區間[l, r]內的最大值、最小值等
//dp[i][j]代表 起始點i, 長度2^j 的區間，轉移方程st[i][j] = max(st[i][j-1], st[i+(1<<(j-1))][j-1])
int n = 101;
vector<int> nums(n);
vector<vector<int>> st(MX, vector<int>(ceil(log2(MX))+1, 0));
int query(int l, int r){
    int j = log2(r-l+1);
    return max(st[l][j], st[r-(1<<j)+1][j]);
}
//base case
for(int i=0;i<n;i++)st[i][0] = nums[i];
//dp
for(int j=1 ; j<=log2(n) ; j++){
    for(int i=0 ; i+(1<<(j-1))<n ; i++){
        st[i][j] = max(st[i][j-1], st[i+(1<<(j-1))][j-1]);
    }
}

//cout<<query(l, r);`
    },
    {
        title: "二分搜索 Binary Search",
        category: "演算法",
        code: `//二分搜搜索 三種寫法 找第一個 ( >= x ) 的 index
vector<int> nums;
sort(nums.begin(), nums.end());
int n = nums.size();
//假設 x 是 3
//閉區間
/*
    l
1 2 3 4 5
  r
*/
int left = 0, right = n-1;
while(left <= right){
    int mid = left + (right - left) / 2;//同 int mid = (left + right) / 2; 但可能overflow
    if(nums[mid] < x){
        left = mid + 1;
    }else{
        right = mid - 1;
    }
}
return left;
//半開區間(左閉右開)
/*
    l
1 2 3 4 5
    r
*/
int left = 0, right = n;
while(left < right){
    int mid = left + (right - left) / 2;//同 int mid = (left + right) / 2; 但可能overflow
    if(nums[mid] < x){
        left = mid + 1;
    }else{
        right = mid;
    }
}
return left;//or return right;
//開區間
/*
  l
1 2 3 4 5
    r
*/
int left = -1, right = n;
while(left + 1 < right){
    int mid = left + (right - left) / 2;//同 int mid = (left + right) / 2; 但可能overflow
    if(nums[mid] < x){
        left = mid;
    }else{
        right = mid;
    }
}
return right;

//leetcode 二分 開區間模板
int left = -1, right = n;//(-1, n) 
auto check = [&](int x){
    //pass
};
while(left+1 < right){
    int mid = left + (right - left) / 2;
    (check(mid) ? right : left) = mid; //(最大值 最小化) FFFFFTTTTT
  //(check(mid) ? left : right) = mid; //(最小值 最大化) TTTTTFFFFF
}
return right;
//return left;

//庫函數 左閉右開
lower_bound(nums.begin(), nums.end(), target);//找第一個 >= target, 減 1 變成 找最後一個 < target的number
upper_bound(nums.begin(), nums.end(), target);//找第一個 > target, 減 1 變成 找最後一個 <= target的number`
    },
    {
        title: "數位DP",
        category: "演算法",
        code: `//小於等於 N(上界)
string s = to_string(n);
int m = s.length();
int dp[10];
memset(dp, -1, sizeof(dp));
auto dfs = [&](this auto&& dfs, int i, bool isHigh, bool isNum){
    if(i == m)return isNum;
    //前面被限制過，沒必要被記憶。
    if(!isHigh && isNum && dp[i]!=-1)return dp[i];
    int res = 0;
    //前面跳過。
    if(!isNum)res = dfs(i+1, false, false);
    
    int hi = (isHigh) ? s[i]-'0' : 9;
    int d0 = (isNum) ? 0 : 1;

    for(int j=0;j<digits.size();j++){
        int x = stoi(digits[j]);
        if(d0<=x && x<=hi){
            res += dfs(i+1, isHigh&&(x==hi), true);
        }
    }
    if(!isHigh && isNum)return dp[i] = res;
    return res;
};
dfs(0, true, false);`
    },
    {
        title: "單調棧 - 找左右最近元素",
        category: "演算法",
        code: `/*
如果要找最近 且 嚴格大於的話 => 代表只要當前數值 大於等於 nums[stk.back()]就要pop() ， pop()完之後，x < nums[stk.back()] 

嚴格< : nums[stk.back()] >= x
嚴格> : nums[stk.back()] <= x
嚴格<= : nums[stk.back()] > x
嚴格>= : nums[stk.back()] < x
*/
vector<int> nums = {1, 5, 1, 4, 5, 1, 3};
int n = nums.size();

vector<int> left(n);//左邊最近 且 嚴格大於
vector<int> stk = {-1};//存下標
for(int i=0;i<n;i++){
    int x = nums[i];
    while(stk.size()>1 && nums[stk.back()] <= x)stk.pop_back();
    left[i] = stk.back();
    stk.push_back(i);
}

vector<int> right(n);
stk = {n};
for(int i=n-1;i>=0;i--){
    int x = nums[i];
    while(stk.size()>1 && nums[stk.back()] <= x)stk.pop_back();
    right[i] = stk.back();
    stk.push_back(i);
}`
    },
    {
        title: "KMP 字串匹配",
        category: "演算法",
        code: `//LeetCode 28. Find the Index of the First Occurrence in a String
int strStr(string text, string pattern) {
    int n = text.length(), m = pattern.length();
    vector<int> lcp(m, 0);
    //lcp[i] 代表 在位置i 能匹配多少長度的前綴
    //a b c a b a b
    //0 0 0 1 2 1 2

    //算法加速的關鍵 => j = lcp[j-1]
    //a b c d a b c z a b c k
    //0 0 0 0 1 2 3 0 1 2 3 0
    //              j       i
    //若 s[i], s[j]無法匹配，j不要直接從頭開始，可以掙扎一下。
    //如果j不為0，代表s[j-1]匹配成功，lcp[j-1]的位置，就是可以掙扎的點。思考上面的範例。
    for(int i=1,j=0;i<m;i++){
        char b = pattern[i];
        while(j && b!=pattern[j])j = lcp[j-1];
        if(b == pattern[j])j++;
        lcp[i] = j;
    }

    for(int i=0,j=0;i<n;i++){
        char b = text[i];
        while(j && b!=pattern[j])j = lcp[j-1];
        if(b == pattern[j])j++;
        if(j == m)return i-m+1;
    }
    return -1;
}`
    },
    {
        title: "Z Function 字串匹配",
        category: "演算法",
        code: `//LeetCode 28. Find the Index of the First Occurrence in a String
int strStr(string text, string pattern) {
    int n = text.length(), m = pattern.length(), N = n+m, left = 0, right = 0;
    pattern.append(text);
    vector<int> z(N, 0);
    for(int i=1;i<N;i++){
        //z[i-left] 表示 s[i-left] 開始的 後綴 能匹配多少長度的 前綴。
        //right - i + 1 表示 s[i] 開始的後綴 "最多"能免費匹配 多少長度的 前綴。
        
        //s[0]
        // |  
        //[i   ...   R     ] => z[i-left]
        //[i   ...   R]      => right - i + 1
        if(i <= right)
            //以z[i - left] 為主，若 z[i - left] <= right-i+1 ，選 z[i - left]
            //                   若 z[i - left] > right-i+1 ，選 right-i+1
            //=> *選 min(z[i-left], right-i+1)
            z[i] = min(z[i-left], right-i+1);

        while(i+z[i]<N && pattern[z[i]]==pattern[i+z[i]]){
            left = i, right = i+z[i];
            z[i]++;
        }
        if(i>=m && z[i]>=m)return i-m;
    }
    return -1;
}`
    },
    {
        title: "不同相鄰元素貪心",
        category: "演算法",
        code: `設 array 中，數組長度為n, 出現頻率最多為 m 次。

問1、給定一個 array, 能否使得相鄰元素均不相同
1 1 2 2 3 -> 1 2 1 2 3
思考點 : 隔一個空位放數字 => 1 _ 1 _ _ => 1 2 1 2 _ => 1 2 1 2 3
結論 : m > ceil(n/2) 就無法，小於等於 則 可以。
/***************************************************************** */
//刪除結束後，一定剩 0或1 個

問2、給定一個array,一次操作中可以刪除兩個不同元素。問:"最多"能操作多少次?
0 0 0 1 1 1 1 2 2
delete (0, 1) * 3 -> delete (1, 2) * 1 => 共 4 次 (floor(n/2)), 若 m < ceil(n/2) , 剩一個的話 不能刪，所以下取整

0 0 0 0 0 1 2
delete (0, 1) -> delete(0, 2) => 共 2 次 (n - m), 若 m >= ceil(n/2)
結論 : min(floor(n/2), n-m) *****
/***************************************************************** */
問3、給定一個array,一次操作中可以刪除至多兩個不同元素。問:"最少"需要操作多少次才能清空數組?
2335. Minimum Amount of Time to Fill Cups

0 0 0 0 1 1 2
delete (0, 1) -> delete (0, 1) -> delete (0, 1) -> delete (0, 2) -> delete 0 => 共 4 次 ， 剩一個的話 必需刪，所以上取整
結論 : max(ceil(n/2), m) *****`
    },
    {
        title: "前綴和 & 後綴和",
        category: "資料結構",
        code: `//vector<int> nums; 
//int n = nums.size();

//前綴和
vector<int> pref(n+1, 0);
for(int i=0;i<n;i++)pref[i+1] = pref[i] + nums[i];//[l, r] = pref[r+1] - pref[l];

//後綴和
vector<int> suff(n+1, 0);
for(int i=0;i<n;i++)suff[i] = suff[i+1] + nums[i];//suff[i] = [i ~ n-1]

//二維前綴和
//pref[i][j] 代表 以 (0, 0), (i, j) 為對角線的長方形面積
/*
1 1 1    1 2 3
1 1 1 -> 2 4 6
1 1 1    3 6 9
*/
int m = grid.size(), n = grid[0].size();
vector<vector<int>> pref(m+1, vector<int>(n+1, 0));
for(int i=0;i<m;i++){
    for(int j=0;j<n;j++){
        pref[i+1][j+1] = pref[i][j+1] + pref[i+1][j] - pref[i][j] + grid[i][j];
    }
}
//計算 以左上角(a, b), 右下角(c, d) 為對角線的長方形面積
/*
#

     #
*/
pref[c+1][d+1] - pref[c+1][b] - pref[a][d+1] + pref[a][b];`
    },
    {
        title: "動態開點線段樹",
        category: "資料結構",
        code: `class segmentTree{
	struct Node{
		int mx = 0;
		Node* left = NULL;
		Node* right = NULL;
	};
	void maintain(Node* o){
		int lmx = o->left ? o->left->mx : 0;
        int rmx = o->right ? o->right->mx : 0;
        o->mx = max(lmx, rmx);
	}
	void update(Node* o, int l, int r, int x){
		if(l == r){
			o->mx = x;
			return ;
		}
		int m = l + (r - l) / 2;
		if(m >= x){
			if(!o->left)o->left = new Node();//沒有就新增
			update(o->left, l, m, x);
		}
		if(m < x){
			if(!o->right)o->right = new Node()//沒有就新增
			update(o->right, m+1, r, x);
		}
		maintain(o);
	}
	int query(Node* o, int l, int r, int ql, int qr){
		if(o==NULL || r<ql || l>qr)return 0;
		if(ql<=l && r<=qr)return o->mx;
		int m = l + (r - l) / 2;
		return max(
			query(o->left, l, m, ql, qr),
			query(o->right, m+1, r, ql, qr)
		);
	}
public:
	int MN,MX;//上下界
	Node* root = new Node();

	segmentTree(int min, int max):MN(min),MX(max){}
	void update(int x){
		update(root, MN, MX, x);
	}
	int query(int ql, int qr){
		return query(root, MN, MX, ql, qr);
	}
};
/*
segmentTree t(mn, mx);
t.update(x);
t.query(l, r);
*/`
    },
    {
        title: "字典樹 Trie",
        category: "資料結構",
        code: `struct Node{
    bool isWord = false;
    Node* child[26];
};
class Trie {
public:
    Node* root;
    Trie() {
        root = new Node();
    }
    
    void insert(string word) {
        Node* ptr = root;
        for(const char& c : word){
            if(!ptr->child[c-'a'])ptr->child[c-'a'] = new Node();
            ptr = ptr->child[c-'a'];
        }
        ptr->isWord = true;
    }
    
    bool search(string word) {
        Node* ptr = root;
        for(const char& c : word){
            if(!ptr->child[c-'a'])return false;
            ptr = ptr->child[c-'a'];
        }
        return ptr->isWord;
    }
    
    bool startsWith(string prefix) {
        Node* ptr = root;
        for(const char& c : prefix){
            if(!ptr->child[c-'a'])return false;
            ptr = ptr->child[c-'a'];
        }
        return true;
    }
};`
    },
    {
        title: "差分數組",
        category: "資料結構",
        code: `//一維差分 
/*
對數組中[l, r]全部 "+1"
對[1, 3] 增加1, 0 0 0 0 0 -> 0 1 1 1 0
差分 -> 0 1 0 0 -1 前綴和還原 0 1 1 1 0 
*/
diff[l]++;
diff[r+1]--;

//二維差分
/*
對 以左上角(a, b) 右下角(c, d) 為對角線的長方形區域 "+1"
0 0 0 0     0 0 0 0
0 0 0 0     0 1 1 0
0 0 0 0  -> 0 1 1 0
0 0 0 0     0 0 0 0

pref[i][j]+1 代表 從 (i, j) 到 (m-1, n-1) 的面積 "都+1"
1. 0 0 0 0   2. 0 0 0 0   3. 0 0 0 0   4. 0 0 0 0
   0 1 0 0      0 1 0 0      0 1 0 -1     0 1 0 -1
   0 0 0 0      0 0 0 0      0 0 0 0      0 0 0 0
   0 0 0 0      0 -1 0 0     0 -1 0 0     0 -1 0 1

=>面積示意圖
1. 0 0 0 0   2. 0 0 0 0   3. 0 0 0 0   4. 0 0 0 0
   0 1 1 1      0 1 1 1      0 1 1 0      0 1 1 0
   0 1 1 1      0 1 1 1      0 1 1 0      0 1 1 0
   0 1 1 1      0 0 0 0      0 0 0 -1     0 0 0 0
*/
pref[a+1][b+1]++;//1
pref[c+2][b+1]--;//2
pref[a+1][d+2]--;//3
pref[c+2][d+2]++;//4`
    },
    {
        title: "懶線段樹 Lazy Segment Tree",
        category: "資料結構",
        code: `class SegmentTree{
    int n;
    vector<int> todo;
    vector<int> t;
    int merge_val(int a, int b){
        return a + b;
    }
    void maintain(int o){
        t[o] = merge_val(t[o*2], t[o*2+1]);
    }
    void build(const vector<int>& nums, int o, int l, int r){
        if(l == r){
            t[o] = nums[l];
            return ;
        }
        int m = l + (r - l) / 2;
        build(nums, o*2, l, m);
        build(nums, o*2+1, m+1, r);
        maintain(o);
    }
    void do_(int o, int l, int r, int val){
        t[o] += (r-l+1) * val;
        todo[o] = val;//lazy tag
    }
    int query(int o, int l, int r, int ql, int qr){
        if(ql<=l && r<=qr){
            return t[o];
        }
        int m = l + (r - l) / 2;
        if(todo[o]){
            do_(o*2, l, m, todo[o]);
            do_(o*2+1, m+1, r, todo[o]);
            todo[o] = 0;
        }
        if(m >= qr)
            return query(o*2, l, m, ql, qr);
        if(m < ql)
            return query(o*2+1, m+1, r, ql, qr);

        return merge_val(
            query(o*2, l, m, ql, qr),
            query(o*2+1, m+1, r, ql, qr)
        );
    }
    void update(int o, int l, int r, int ql, int qr, int val){
        if(ql<=l && r<=qr){
            do_(o, l, r, val);
            return ;
        }
        int m = l + (r - l) / 2;
        if(todo[o]){
            do_(o*2, l, m, todo[o]);
            do_(o*2+1, m+1, r, todo[o]);
            todo[o] = 0;
        }
        if(m >= ql)
            update(o*2, l, m, ql, qr, val);
        if(m < qr)
            update(o*2+1, m+1, r, ql, qr, val);
        maintain(o);
    }
public:
    SegmentTree(const vector<int>& nums){
        n = nums.size();
        t.assign(4*n, 0);
        todo.assign(4*n, 0);
        build(nums, 1, 0, n-1);
    }
    void update(int l, int r, int val){
        return update(1, 0, n-1, l, r, val);
    }
    int query(int l, int r){
        return query(1, 0, n-1, l, r);
    }
};
/*
t.update(l, r, val); 把[l, r] 增加 val
t.query(l, r); 求區間[l, r] 的 sum
*/`
    },
    {
        title: "樹狀數組 Fenwick Tree",
        category: "資料結構",
        code: `class FenWick{
    vector<int> t;
    int sz;
public:
    FenWick(int n):t(n+1), sz(n){}
    void update(int x, int add){
        int i = x;
        while(i <= sz){
            t[i] += add;
            i += i&-i;//加上 low bit
        }
    }
    //找 [1 ~ x] 的和 (1 - index)
    int pre(int x){
        int sum = 0, i = x;
        while(i > 0){
            sum += t[i];
            i -= i&-i;//減去 low bit
        }
        return sum;
    }
    //找 [l, r] 的和 (0 - index)
    int query(int l, int r){
        return pre(r+1) - pre(l);
    }
};
/*
t.update(i+1, 1); => 更新 nums[i]
t.pre(i+1); => 計算 nums[0 ~ i] 的和
t.query(l, r) => 計算 nums[l ~ r] 的和
*/`
    },
    {
        title: "線段樹 Segment Tree",
        category: "資料結構",
        code: `class SegmentTree{
    int n;
    vector<int> mx;
    int merge_val(int a, int b){
        return max(a, b);
    }
    void maintain(int o){
        mx[o] = merge_val(mx[o*2], mx[o*2+1]);
    }
    void build(const vector<int>& nums, int o, int l, int r){
        if(l == r){
            mx[o] = nums[l];
            return ;
        }
        int m = l + (r - l) / 2;
        build(nums, o*2, l, m);
        build(nums, o*2+1, m+1, r);
        maintain(o);
    }
    int query(int o, int l, int r, int ql, int qr){
        if(ql<=l && r<=qr){
            return mx[o];
        }
        int m = l + (r - l) / 2;
        if(m >= qr)
            return query(o*2, l, m, ql, qr);
        if(m < ql)
            return query(o*2+1, m+1, r, ql, qr);

        return merge_val(
            query(o*2, l, m, ql, qr),
            query(o*2+1, m+1, r, ql, qr)
        );
    }
    void update(int o, int l, int r, int i, int val){
        if(l == r){
            mx[o] = val;
            return ;
        }
        int m = l + (r - l) / 2;
        if(m >= i)
            update(o*2, l, m, i, val);
        else
            update(o*2+1, m+1, r, i, val);
        maintain(o);
    }
    int findFirst(int o, int l, int r, int val){
        if(mx[o] < val)return -1;
        if(l == r)return l;

        int m = l + (r - l) / 2;
        int i = findFirst(o*2, l, m, val);
        if(i < 0)
            i = findFirst(o*2+1, m+1, r, val);
        return i;
    }
public:
    SegmentTree(const vector<int>& nums){
        n = nums.size();
        mx.resize(4*n);
        build(nums, 1, 0, n-1);
    }
    void update(int i, int val){
        update(1, 0, n-1, i, val);
    }
    int query(int l, int r){
        return query(1, 0, n-1, l, r);
    }
    int findFirst(int val){
        return findFirst(1, 0, n-1, val);
    }
};
/*
t.query(l, r);求區間[l, r] 的 最大值(or 和、最小值等)
t.update(i, val);把 nums[i] 更新成 val
t.findFirst(val);找第一個值 >= 為val 的 index
*/`
    }
];

// Extract categories
const uniqueCategories = [...new Set(codeData.map(item => item.category))];
const categories = ['all', ...uniqueCategories];

// Render category buttons
const categoryNav = document.getElementById('categoryNav');
categoryNav.innerHTML = ''; // Clear existing buttons
categories.forEach(cat => {
    const btn = document.createElement('button');
    btn.className = 'category-btn';
    if (cat === 'all') btn.classList.add('active');
    btn.textContent = cat === 'all' ? '全部' : cat;
    btn.dataset.category = cat;
    btn.onclick = () => filterByCategory(cat);
    categoryNav.appendChild(btn);
});

// Render code cards
function renderCards(data) {
    const grid = document.getElementById('contentGrid');
    const noResults = document.getElementById('noResults');

    if (data.length === 0) {
        grid.style.display = 'none';
        noResults.style.display = 'block';
        return;
    }

    grid.style.display = 'grid';
    noResults.style.display = 'none';
    grid.innerHTML = '';

    data.forEach((item, index) => {
        const card = document.createElement('div');
        card.className = 'code-card';
        card.style.animationDelay = `${index * 0.1}s`;

        card.innerHTML = `
            <div class="card-header">
                <div class="card-title">${item.title}</div>
                <div class="category-tag">${item.category}</div>
            </div>
            <div class="card-body">
                <button class="copy-btn" onclick="copyCode(this, ${index})">📋 複製</button>
                <pre><code class="language-cpp">${escapeHtml(item.code)}</code></pre>
            </div>
        `;

        grid.appendChild(card);
    });

    // Highlight code
    document.querySelectorAll('pre code').forEach((block) => {
        hljs.highlightElement(block);
    });
}

// Filter by category
function filterByCategory(category) {
    // Update active button
    document.querySelectorAll('.category-btn').forEach(btn => {
        btn.classList.remove('active');
        if (btn.dataset.category === category) {
            btn.classList.add('active');
        }
    });

    // Filter data
    const filtered = category === 'all'
        ? codeData
        : codeData.filter(item => item.category === category);

    renderCards(filtered);
}

// Search functionality
const searchInput = document.getElementById('searchInput');
searchInput.addEventListener('input', (e) => {
    const query = e.target.value.toLowerCase();
    const filtered = codeData.filter(item =>
        item.title.toLowerCase().includes(query) ||
        item.category.toLowerCase().includes(query) ||
        item.code.toLowerCase().includes(query)
    );
    renderCards(filtered);
});

// Copy code function
window.copyCode = function (btn, index) {
    const code = codeData[index].code;
    navigator.clipboard.writeText(code).then(() => {
        btn.textContent = '✓ 已複製';
        btn.classList.add('copied');
        setTimeout(() => {
            btn.textContent = '📋 複製';
            btn.classList.remove('copied');
        }, 2000);
    });
};

// Modal functions
window.openModal = function () {
    document.getElementById('addCodeModal').classList.add('active');
    document.body.style.overflow = 'hidden';
};

window.closeModal = function () {
    document.getElementById('addCodeModal').classList.remove('active');
    document.body.style.overflow = 'auto';
    document.getElementById('addCodeForm').reset();
};

// Handle form submission
window.handleSubmit = function (event) {
    event.preventDefault();

    const title = document.getElementById('codeTitle').value;
    const category = document.getElementById('codeCategory').value;
    const code = document.getElementById('codeContent').value;

    // Add new code to codeData
    codeData.push({
        title: title,
        category: category,
        code: code
    });

    // Save to localStorage
    localStorage.setItem('customCodes', JSON.stringify(codeData));

    // Update categories
    updateCategories();

    // Re-render cards
    const currentCategory = document.querySelector('.category-btn.active').dataset.category;
    filterByCategory(currentCategory);

    // Close modal
    closeModal();

    // Show success message
    alert('✅ 代碼已成功新增！');
};

// Update categories after adding new code
function updateCategories() {
    const uniqueCategories = [...new Set(codeData.map(item => item.category))];
    const categories = ['all', ...uniqueCategories];

    const categoryNav = document.getElementById('categoryNav');
    const activeCategory = document.querySelector('.category-btn.active')?.dataset.category || 'all';
    categoryNav.innerHTML = '';

    categories.forEach(cat => {
        const btn = document.createElement('button');
        btn.className = 'category-btn';
        if (cat === activeCategory) btn.classList.add('active');
        btn.textContent = cat === 'all' ? '全部' : cat;
        btn.dataset.category = cat;
        btn.onclick = () => filterByCategory(cat);
        categoryNav.appendChild(btn);
    });
}

// Close modal when clicking outside
document.getElementById('addCodeModal').addEventListener('click', function (e) {
    if (e.target === this) {
        closeModal();
    }
});

// Load custom codes from localStorage
function loadCustomCodes() {
    const saved = localStorage.getItem('customCodes');
    if (saved) {
        try {
            const parsed = JSON.parse(saved);
            // Only load custom codes that aren't in the original data
            const originalTitles = codeData.map(item => item.title);
            const customCodes = parsed.filter(item => !originalTitles.includes(item.title));
            codeData.push(...customCodes);
        } catch (e) {
            console.error('Error loading custom codes:', e);
        }
    }
}

// Escape HTML
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Initial render
loadCustomCodes();
renderCards(codeData);