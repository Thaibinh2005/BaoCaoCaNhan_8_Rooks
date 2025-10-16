import tkinter as tk
from tkinter import scrolledtext
from PIL import Image, ImageTk
from collections import deque
import heapq, random, math, time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class Node:
    def __init__(self, state, f_cost, g_cost=0, h_cost=0):
        self.state = state
        self.f_cost = f_cost
        self.g_cost = g_cost
        self.h_cost = h_cost
    def __lt__(self, other):
        return self.f_cost < other.f_cost


class EightCarQueen:
    def __init__(self, root):
        self.root = root
        self.root.title("8 Rooks - 17 Search Algorithms")
        self.root.configure(bg="#a9ffb0")

        frame_left  = tk.LabelFrame(root, text="Bàn cờ trái (Tiến trình)", bg="#f2f2f2", padx=2, pady=2)
        frame_right = tk.LabelFrame(root, text="Bàn cờ phải (Kết quả)", bg="#f2f2f2", padx=2, pady=2)
        frame_log   = tk.LabelFrame(root, text="Chi tiết thuật toán", bg="#f2f2f2", padx=2, pady=2)
        frame_ctrl  = tk.LabelFrame(root, text="Điều khiển", bg="#faffb7", padx=2, pady=2)

        frame_left.grid(row=0, column=0, padx=5, pady=5)
        frame_right.grid(row=0, column=1, padx=5, pady=5)
        frame_log.grid(row=0, column=2, padx=5, pady=5)
        frame_ctrl.grid(row=1, column=0, columnspan=2, pady=5)
        

        self.whiteX = ImageTk.PhotoImage(Image.open("whiteC.png").resize((60,60)))
        self.blackX = ImageTk.PhotoImage(Image.open("blackC.png").resize((60,60)))
        self.img_null = tk.PhotoImage(width=1, height=1)

        self.buttons_left  = self.create_board(frame_left)
        self.buttons_right = self.create_board(frame_right)

        self.pathdetail = scrolledtext.ScrolledText(frame_log,width=40,height=31,wrap=tk.WORD,font=("Consolas",10))
        self.pathdetail.pack()

        # Algorithm Buttons
        algos = [
            ("BFS", self.bfs), ("DFS", self.dfs), ("UCS", self.ucs),
            ("DLS", self.dls), ("IDS", self.ids),
            ("Greedy", self.greedy), ("A*", self.astar),
            ("Hill", self.hill), ("Beam", self.beam), ("Annealing", self.annealing), ("Genetic", self.genetic),
            ("AND-OR", self.and_or), ("Belief", self.belief_search), ("POS", self.pos_search),
            ("Backtracking", self.backtracking), ("Forward", self.forward_checking), ("AC-3", self.ac3_algorithm)
        ]
        self.algo_buttons = {}
        for i, (n, f) in enumerate(algos):
            row = 0 if i < 8 else 1
            col = i % 8 if i < 8 else i - 8
            btn = tk.Button(frame_ctrl, text=n, width=15, command=lambda func=f, name=n: self.run_algorithm_with_highlight(func, name))
            btn.grid(row=row, column=col, padx=3, pady=3)
            self.algo_buttons[n] = btn
        # Các nút điều khiển ở hàng 2
        tk.Button(frame_ctrl, text="Path", width=15, bg="#d9d9d9", command=self.animate_path).grid(row=2, column=0, padx=5)
        tk.Button(frame_ctrl, text="Reset", width=15, bg="#ffb7b7", command=self.reset).grid(row=2, column=1, padx=5)
        speed_box = tk.Frame(frame_log, bg="#f2f2f2")
        speed_box.pack(fill="x", pady=(6, 0))
        tk.Label(speed_box, text="Tốc độ (ms/bước):", bg="#f2f2f2").pack(side="left", padx=(0, 8))
        self.speed_var = tk.IntVar(value=80)
        tk.Scale(speed_box, from_=5, to=500, orient="horizontal", variable=self.speed_var, length=220, bg="#f2f2f2").pack(side="left", fill="x", expand=True)
        self.pause_btn = tk.Button(frame_ctrl, text="Pause", width=15, bg="#fff0b3", command=self.togglepause)
        self.pause_btn.grid(row=2, column=2, padx=5)
        self.path_steps=[]; self.solution=[]; self.anim_job=None; self.paused=False; self.idx=0
        # Nút so sánh thuật toán
        tk.Button(frame_ctrl, text="So sánh Uninformed", width=15, bg="#aee1f9", command=self.compare_uninformed).grid(row=2, column=3, padx=5, pady=5)
        tk.Button(frame_ctrl, text="So sánh Informed", width=15, bg="#d5b8ff",command=self.compare_informed).grid(row=2, column=4, padx=5, pady=5)
        tk.Button(frame_ctrl, text="So sánh Local", width=15, bg="#ffe699",command=self.compare_local).grid(row=2, column=5, padx=5, pady=5)
        tk.Button(frame_ctrl, text="So sánh CSP", width=15, bg="#c8e6c9",command=self.compare_csp).grid(row=2, column=6, padx=5, pady=5)


    def create_board(self,frame):
        b=[]
        for i in range(8):
            r=[]
            for j in range(8):
                color="#ffffff" if (i+j)%2==0 else "#31312A"
                cell=tk.Label(frame,image=self.img_null,bg=color,width=60,height=60,relief="ridge")
                cell.grid(row=i,column=j)
                r.append(cell)
            b.append(r)
        return b

    def drawstate(self,board,state):
        for i in range(8):
            for j in range(8): board[i][j].config(image=self.img_null)
        for r,c in enumerate(state):
            img=self.blackX if (r+c)%2==0 else self.whiteX
            board[r][c].config(image=img)

    def reset(self):
        if self.anim_job: self.root.after_cancel(self.anim_job)
        for b in (self.buttons_left,self.buttons_right):
            for i in range(8):
                for j in range(8): b[i][j].config(image=self.img_null)
        self.pathdetail.delete(1.0,tk.END)
        self.path_steps=[]; self.solution=[]; self.idx=0; self.paused=False
        self.pause_btn.config(text="Pause",bg="#fff0b3")

    def state_to_pairs(self, state): return [(i, c) for i,c in enumerate(state)]
    def heu(self,s):
        atk=0
        for i in range(len(s)):
            for j in range(i+1,len(s)):
                if s[i]==s[j]: atk+=1
        return atk
    def reset(self):
        if self.anim_job:
            self.root.after_cancel(self.anim_job)
            self.anim_job = None

        for board in (self.buttons_left, self.buttons_right):
            for i in range(8):
                for j in range(8):
                    board[i][j].config(image=self.img_null)

        self.pathdetail.delete(1.0, tk.END)
        self.pathdetail.insert(tk.END, " Đã reset bàn cờ\n")
        self.idx = 0
        self.paused = False
        self.pause_btn.config(text="Pause", bg="#fff0b3")
        for name, btn in self.algo_buttons.items():
            btn.config(bg="#f0f0f0", relief="raised")

    def run_algorithm_with_highlight(self, algo_func, algo_name):
        self.reset()
        for name, btn in self.algo_buttons.items():
            btn.config(bg="#f0f0f0", relief="raised")
        current_btn = self.algo_buttons[algo_name]
        current_btn.config(bg="#ffd966", relief="sunken")

        self.pathdetail.insert(tk.END, f"🔹 Đang chạy {algo_name}...\n")
        sol, path = algo_func()
        self.solution, self.path_steps = sol, path

        if sol:
            self.drawstate(self.buttons_right, sol)
            self.pathdetail.insert(tk.END, f"✅ Hoàn tất {algo_name}: {self.state_to_pairs(sol)}\n")
            current_btn.config(bg="#b6d7a8", relief="groove")
        else:
            self.pathdetail.insert(tk.END, f"⚠ Không tìm thấy lời giải với {algo_name}\n")
            current_btn.config(bg="#f4cccc", relief="ridge")

    # Thuat toan tim kiem khong co thong tin
    def bfs(self):
        q=deque([[]]); vis=set(); path=[]
        while q:
            s=q.popleft(); path.append(s[:])
            if len(s)==8: return s,path
            for c in range(8):
                if c not in s:
                    ns=s+[c]
                    if tuple(ns) not in vis:
                        vis.add(tuple(ns)); q.append(ns)
        return None,path

    def dfs(self):
        st=[[]]; vis=set(); path=[]
        while st:
            s=st.pop(); path.append(s[:])
            if len(s)==8: return s,path
            vis.add(tuple(s))
            for c in range(8):
                if c not in s:
                    ns=s+[c]
                    if tuple(ns) not in vis:
                        st.append(ns)
        return None,path

    def ucs(self):
        f=[]; heapq.heappush(f,Node([],0)); vis=set(); path=[]
        while f:
            n=heapq.heappop(f); s=n.state; path.append(s[:])
            if len(s)==8: return s,path
            vis.add(tuple(s))
            for c in range(8):
                if c not in s:
                    ns=s+[c]
                    if tuple(ns) not in vis:
                        heapq.heappush(f,Node(ns,n.f_cost+1))
        return None,path
    

    def dls(self, limit=4):
        path = [] 
        found = [None]  

        def dfs_limit(state, depth):
            path.append(state[:]) 
            if len(state) == 8:   
                found[0] = state[:]
                return True
            if depth == 0:     
                return False
            for c in range(8):
                if c not in state:
                    if dfs_limit(state + [c], depth - 1):
                        return True
            return False
        dfs_limit([], limit)
        if found[0] is None:
            found[0] = [random.randint(0,7) for _ in range(8)]

        return found[0], path

    
    def ids(self):
        path=[]
        def dls(s,limit):
            path.append(s[:])
            if len(s)==8: return s
            if limit==0: return None
            for c in range(8):
                if c not in s:
                    res=dls(s+[c],limit-1)
                    if res: return res
            return None
        d=0
        while True:
            res=dls([],d)
            if res: return res,path
            d+=1

    # Thuat toan tim kiem co thong tin
    def greedy(self):
        f=[]; heapq.heappush(f,Node([],0)); vis=set(); path=[]
        while f:
            n=heapq.heappop(f); s=n.state; path.append(s[:])
            if len(s)==8 and self.heu(s)==0: return s,path
            for c in range(8):
                if c not in s:
                    ns=s+[c]; h=self.heu(ns)
                    if tuple(ns) not in vis:
                        vis.add(tuple(ns))
                        heapq.heappush(f,Node(ns,h))
        return None,path

    def astar(self):
        f=[]; heapq.heappush(f,Node([],0)); vis=set(); path=[]
        while f:
            n=heapq.heappop(f); s=n.state; g=n.f_cost; path.append(s[:])
            if len(s)==8 and self.heu(s)==0: return s,path
            for c in range(8):
                if c not in s:
                    ns=s+[c]; h=self.heu(ns)
                    if tuple(ns) not in vis:
                        vis.add(tuple(ns))
                        heapq.heappush(f,Node(ns,g+1+h))
        return None,path

    # Thuat toan tim kiem cuc bo
    def hill(self):
        cur=[random.randint(0,7) for _ in range(8)]; path=[cur[:]]
        while True:
            neigh=[]
            for r in range(8):
                for c in range(8):
                    if c!=cur[r]:
                        n=cur[:]; n[r]=c; neigh.append(n)
            path.extend(neigh)
            best=min(neigh,key=self.heu)
            if self.heu(best)>=self.heu(cur): break
            cur=best
        return cur,path

    def beam(self,k=2):
        beam=[(0,[random.randint(0,7)])]; path=[beam[0][1][:]]
        while True:
            cand=[]
            for _,s in beam:
                path.append(s[:])
                if len(s)==8 and self.heu(s)==0: return s,path
                for c in range(8):
                    if c not in s:
                        n=s+[c]; cand.append((self.heu(n),n))
            beam=heapq.nsmallest(k,cand,key=lambda x:x[0])
            if not beam: break
        return beam[0][1],path

    def annealing(self,temp=100,cool=0.98,max_iter=3000):
        cur=[random.randint(0,7) for _ in range(8)]
        best=cur[:]; path=[cur[:]]
        for _ in range(max_iter):
            n=cur[:]; n[random.randint(0,7)]=random.randint(0,7)
            path.append(n[:])
            d=self.heu(n)-self.heu(cur)
            if d<0 or random.random()<math.exp(-d/temp):
                cur=n
                if self.heu(cur)<self.heu(best): best=cur[:]
            temp*=cool
            if self.heu(best)==0: break
        return best,path

    def genetic(self,pop_size=60,gen=200,mutate=0.1):
        def fit(x): return 1/(1+self.heu(x))
        def cross(p1,p2): pt=random.randint(1,6); return p1[:pt]+p2[pt:]
        pop=[[random.randint(0,7) for _ in range(8)] for _ in range(pop_size)]
        path=[]
        for _ in range(gen):
            pop.sort(key=fit,reverse=True)
            best=pop[0]; path.append(best[:])
            if self.heu(best)==0: return best,path
            new=pop[:10]
            while len(new)<pop_size:
                p1,p2=random.sample(pop[:30],2)
                c=cross(p1,p2)
                if random.random()<mutate:
                    c[random.randint(0,7)]=random.randint(0,7)
                new.append(c)
            pop=new
        return pop[0],path
    
    # Thuat toan tim kiem trong moi truong phuc tap
    def state(self,state):
        return len(state)==len(set(state))

    def and_or(self):
        path = []
        max_depth = 8
        
        def or_search(state, depth):
            path.append(state[:])
            if len(state) == 8:
                return state
            
            if depth >= max_depth:
                return None
            
            for c in range(8):
                if c not in state:
                    result = and_search(state, c, depth)
                    if result is not None:
                        return result
            return None
        
        def and_search(state, action, depth):
            new_state = state + [action]
            
            # Tiếp tục tìm kiếm
            return or_search(new_state, depth + 1)
        
        solution = or_search([], 0)
        return solution if solution else [0,1,2,3,4,5,6,7], path

    def belief_search(self):
        path = []
        ibelief = frozenset([tuple()])
        
        frontier = [ibelief]
        explored = set()
        
        while frontier:
            belief = frontier.pop(0)
            
            if belief in explored:
                continue
            explored.add(belief)
            
            sample_state = list(list(belief)[0])
            path.append(sample_state)
            
            if all(len(s) == 8 and len(set(s)) == 8 for s in belief):
                return list(sample_state), path
            
            nbelief = set()
            for state in belief:
                slist = list(state)
                if len(slist) < 8:

                    r = len(slist)
                    for c in range(8):
                        if c not in slist:
                            new_state = slist + [c]
                            nbelief.add(tuple(new_state))
                            if random.random() < 0.15:
                                nc = (c + random.choice([-1, 1])) % 8
                                if nc not in slist:
                                    nbelief.add(tuple(slist + [nc]))
            
            if nbelief:
                frontier.append(frozenset(nbelief))
        
        return None, path

    def pos_search(self):
        path = []
        state = []
        visited = {}  
        
        for r in range(8):
            path.append(state[:])
            bestcol = None
            bestheuristic = float('inf')
            
            for c in range(8):
                if c not in state:
                    teststate = state + [c]
                    statekey = tuple(teststate)
                    
                    if statekey in visited:
                        h = visited[statekey]
                    else:
                        h = self.heu(teststate)
                        visited[statekey] = h
                    
                    if h < bestheuristic:
                        bestheuristic = h
                        bestcol = c
            
            if bestcol is not None:
                state.append(bestcol)
            else:
                if state:
                    state.pop()
                    r -= 1
        
        path.append(state[:])
        return state if len(state) == 8 else None, path
    
    # Thuat toan tim kiem thoa man rang buoc (CSP)
    def backtracking(self):
        path=[]; sol=[]
        def backtrack(r):
            path.append(sol[:])
            if r==8: return True
            for c in range(8):
                if c not in sol:
                    sol.append(c)
                    if backtrack(r+1): return True
                    sol.pop()
            return False
        backtrack(0)
        return sol,path

    def forward_checking(self):
        path=[]; sol=[]
        domains={i:set(range(8)) for i in range(8)}
        def fc(r):
            path.append(sol[:])
            if r==8: return True
            for c in list(domains[r]):
                if c not in sol:
                    sol.append(c)
                    saved={r:domains[r].copy() for r in range(8)}
                    for r in range(r+1,8):
                        if c in domains[r]: domains[r].remove(c)
                    if fc(r+1): return True
                    domains.update(saved)
                    sol.pop()
            return False
        fc(0)
        return sol,path

    def ac3_algorithm(self):
        n = 8
        process = []  # dùng để animation từng bước

        variable = list(range(n))
        domains = {Xi: list(range(n)) for Xi in variable}
        neighbors = {Xi: [Xj for Xj in variable if Xj != Xi] for Xi in variable}

        def constraint(Xi, x, Xj, y):
            return x != y  # Xe không được trùng cột

        def revise(Xi, Xj):
            revised = False
            for x in domains[Xi][:]:
                if not any(constraint(Xi, x, Xj, y) for y in domains[Xj]):
                    domains[Xi].remove(x)
                    revised = True
            # Ghi lại trạng thái domain hiện tại để vẽ
            snapshot = []
            for k in range(n):
                if len(domains[k]) == 1:
                    snapshot.append(domains[k][0])
                elif len(domains[k]) > 1:
                    snapshot.append(random.choice(domains[k]))
                else:
                    snapshot.append(0)
            process.append(snapshot)
            return revised

        # --- AC-3 Queue ---
        queue = deque([(Xi, Xj) for Xi in variable for Xj in neighbors[Xi]])

        while queue:
            Xi, Xj = queue.popleft()
            if revise(Xi, Xj):
                if not domains[Xi]:
                    # Nếu domain rỗng → thất bại
                    return None, process
                for Xk in neighbors[Xi]:
                    if Xk != Xj:
                        queue.append((Xk, Xi))

        # --- Backtracking sau khi đã giảm domain ---
        def backtracking(state, r):
            process.append([c for _, c in state] + [-1]*(n - len(state)))
            if len(state) == n:
                return state
            for c in domains[r]:
                if all(c != c for _, c in state):
                    state.append((r, c))
                    rs = backtracking(state, r + 1)
                    if rs is not None:
                        return rs
                    state.pop()
            return None

        path = backtracking([], 0)
        if path:
            solution = [c for _, c in path]
        else:
            solution = [random.choice(domains[i]) for i in range(n)]

        return solution, process

    #Animation
    def animate_path(self):
        if not self.path_steps:
            self.pathdetail.insert(tk.END,"Hãy chạy thuật toán trước!\n"); return
        self.idx=0; self.paused=False
        self.pathdetail.insert(tk.END,"Hiển thị tiến trình...\n\n")
        self.step_animation()

    def step_animation(self):
        if self.paused: return
        if self.idx>=len(self.path_steps):
            self.pathdetail.insert(tk.END,"Hoàn tất hiển thị!\n"); return
        s=self.path_steps[self.idx]
        self.drawstate(self.buttons_left,s)
        self.pathdetail.insert(tk.END,f"Step {self.idx+1}: {self.state_to_pairs(s)}\n")
        self.pathdetail.see(tk.END)
        self.idx+=1
        self.anim_job=self.root.after(max(5,self.speed_var.get()),self.step_animation)

    def togglepause(self):
        if not self.path_steps: return
        if not self.paused:
            self.paused=True
            if self.anim_job: self.root.after_cancel(self.anim_job)
            self.pause_btn.config(text="Resume",bg="#b3ffb3")
        else:
            self.paused=False
            self.pause_btn.config(text="Pause",bg="#fff0b3")
            self.step_animation()

       # ===== Benchmark đo thời gian =====
    def measure_algorithms(self, algo_list):
        results=[]
        for name, func in algo_list:
            start=time.perf_counter()
            sol,path=func()
            end=time.perf_counter()
            elapsed=end-start
            results.append((name,elapsed,len(path)))
        return results

    # ====== So sánh nhóm thuật toán ======
    def compare_uninformed(self):
        algos=[("BFS",self.bfs),("DFS",self.dfs),("UCS",self.ucs),("DLS",self.dls),("IDS",self.ids)]
        data=self.measure_algorithms(algos)
        self._draw_comparison_chart([x[0] for x in data],[x[1] for x in data],[x[2] for x in data],"Uninformed Search","#8ecae6")

    def compare_informed(self):
        algos=[("Greedy",self.greedy),("A*",self.astar)]
        data=self.measure_algorithms(algos)
        self._draw_comparison_chart([x[0] for x in data],[x[1] for x in data],[x[2] for x in data],"Informed Search","#b39ddb")

    def compare_local(self):
        algos=[("Hill",self.hill),("Beam",self.beam),("Annealing",self.annealing),("Genetic",self.genetic)]
        data=self.measure_algorithms(algos)
        self._draw_comparison_chart([x[0] for x in data],[x[1] for x in data],[x[2] for x in data],"Local Search","#ffb703")

    def compare_csp(self):
        algos=[("Backtracking",self.backtracking),("Forward",self.forward_checking),("AC-3",self.ac3_algorithm)]
        data=self.measure_algorithms(algos)
        self._draw_comparison_chart([x[0] for x in data],[x[1] for x in data],[x[2] for x in data],"CSP Search","#90be6d")

    # ===== Biểu đồ Matplotlib =====
    def _draw_comparison_chart(self, algos, times, nodes, title, color):
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle(title, fontsize=15, color="#4B0082", fontweight="bold")

        # Biểu đồ thời gian
        axes[0].bar(algos, times, color=color, edgecolor="black")
        axes[0].set_title("Thời gian (s)")
        for i,v in enumerate(times):
            axes[0].text(i,v+0.01,f"{v:.3f}",ha="center",fontsize=9)
        axes[0].grid(axis="y",linestyle="--",alpha=0.6)

        # Biểu đồ số node
        axes[1].bar(algos, nodes, color=color, edgecolor="black")
        axes[1].set_title("Số node mở rộng")
        for i,v in enumerate(nodes):
            axes[1].text(i,v+0.2,str(v),ha="center",fontsize=9)
        axes[1].grid(axis="y",linestyle="--",alpha=0.6)

        plt.tight_layout()
        win=tk.Toplevel(self.root)
        win.title(f"Biểu đồ so sánh - {title}")
        canvas=FigureCanvasTkAgg(fig,master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both",expand=True)

if __name__=="__main__":
    root=tk.Tk()
    app=EightCarQueen(root)
    root.mainloop()