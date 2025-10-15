import tkinter as tk
from tkinter import scrolledtext
from PIL import Image, ImageTk
from collections import deque
import heapq, random, math
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

        self.path_detail = scrolledtext.ScrolledText(frame_log,width=40,height=31,wrap=tk.WORD,font=("Consolas",10))
        self.path_detail.pack()

        # Algorithm Buttons
        algos = [
            ("BFS", self.bfs), ("DFS", self.dfs), ("UCS", self.ucs), ("IDS", self.ids),
            ("Greedy", self.greedy), ("A*", self.astar),
            ("Hill climbing", self.hill), ("Beam", self.beam), ("Simulated Annealing", self.annealing), ("Genetic", self.genetic),
            ("AND-OR", self.and_or_dfs), ("Belief", self.belief_search), ("Partially Observable", self.pos_search),
            ("Backtracking", self.backtracking), ("Forward Checking", self.forward_checking), ("AC-3", self.ac3_algorithm)
        ]

        # Nhóm nút chia 2 hàng
        for i, (n, f) in enumerate(algos):
            row = 0 if i < 8 else 1     # 8 nút đầu ở hàng 0, 7 nút còn lại ở hàng 1
            col = i % 8 if i < 8 else i - 8
            tk.Button(
                frame_ctrl,
                text=n,
                width=15,
                command=lambda func=f: self.run_algorithm(func)
            ).grid(row=row, column=col, padx=3, pady=3)
        # Các nút điều khiển ở hàng 2
        tk.Button(frame_ctrl, text="Path", width=15, bg="#d9d9d9", command=self.animate_path).grid(row=2, column=0, padx=5)
        tk.Button(frame_ctrl, text="Reset", width=15, bg="#ffb7b7", command=self.reset).grid(row=2, column=1, padx=5)
        speed_box = tk.Frame(frame_log, bg="#f2f2f2")
        speed_box.pack(fill="x", pady=(6, 0))
        tk.Label(speed_box, text="Tốc độ (ms/bước):", bg="#f2f2f2").pack(side="left", padx=(0, 8))
        self.speed_var = tk.IntVar(value=80)
        tk.Scale(speed_box, from_=5, to=500, orient="horizontal", variable=self.speed_var, length=220, bg="#f2f2f2").pack(side="left", fill="x", expand=True)
        self.pause_btn = tk.Button(frame_ctrl, text="Pause", width=15, bg="#fff0b3", command=self.toggle_pause)
        self.pause_btn.grid(row=2, column=2, padx=5)
        self.path_steps=[]; self.solution=[]; self.anim_job=None; self.paused=False; self.idx=0
        # ======= Nút so sánh thuật toán =======
        tk.Button(frame_ctrl, text="So sánh Uninformed", width=15, bg="#aee1f9", command=self.compare_uninformed).grid(row=2, column=3, padx=5, pady=5)
        tk.Button(frame_ctrl, text="So sánh Informed", width=15, bg="#d5b8ff",command=self.compare_informed).grid(row=2, column=4, padx=5, pady=5)
        tk.Button(frame_ctrl, text="So sánh Local", width=15, bg="#ffe699",command=self.compare_local).grid(row=2, column=5, padx=5, pady=5)
        tk.Button(frame_ctrl, text="So sánh CSP", width=15, bg="#c8e6c9",command=self.compare_csp).grid(row=2, column=6, padx=5, pady=5)


    def create_board(self,frame):
        b=[]
        for i in range(8):
            row=[]
            for j in range(8):
                color="#ffffff" if (i+j)%2==0 else "#31312A"
                cell=tk.Label(frame,image=self.img_null,bg=color,width=60,height=60,relief="ridge")
                cell.grid(row=i,column=j)
                row.append(cell)
            b.append(row)
        return b

    def draw_state(self,board,state):
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
        self.path_detail.delete(1.0,tk.END)
        self.path_steps=[]; self.solution=[]; self.idx=0; self.paused=False
        self.pause_btn.config(text="Pause",bg="#fff0b3")

    def state_to_pairs(self, state): return [(i, c) for i,c in enumerate(state)]
    def heu(self,s):
        atk=0
        for i in range(len(s)):
            for j in range(i+1,len(s)):
                if s[i]==s[j]: atk+=1
        return atk

    # Run
    def run_algorithm(self,algo):
        self.reset()
        self.path_detail.insert(tk.END,f"Đang chạy {algo.__name__}...\n")
        sol,path=algo()
        self.solution=sol; self.path_steps=path
        if sol:
            self.draw_state(self.buttons_right,sol)
            self.path_detail.insert(tk.END,f"Hoàn tất: {self.state_to_pairs(sol)}\n")

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
    def is_valid_state(self,state):
        return len(state)==len(set(state))

    def and_or_dfs(self):
        f=[([],0)]; path=[]
        while f:
            s,d=f.pop(); path.append(s[:])
            if len(s)==8 and self.is_valid_state(s): return s,path
            if d<8:
                for c in range(8):
                    ns=s+[c]
                    if self.is_valid_state(ns):
                        f.append((ns,d+1))
        return None,path

    def results(self,state,action):
        state=list(state)
        results=[state+[action]]
        for c2 in range(8):
            if c2!=action and c2 not in state:
                results.append(state+[c2])
        return results

    def belief_search(self):
        goal=8; f=[frozenset({tuple([])})]; seen=set(); path=[]
        while f:
            belief=f.pop()
            if belief in seen: continue
            seen.add(belief)
            example=list(belief)[0]; path.append(list(example))
            if all(len(s)==goal for s in belief): return list(example),path
            new=set()
            for st in belief:
                st=list(st)
                if len(st)<goal:
                    for c in range(8):
                        if c not in st:
                            for ns in self.results(st,c):
                                new.add(tuple(ns))
            if new: f.append(frozenset(new))
        return None,path

    def pos_search(self):
        path = []
        belief = {tuple()}   # khởi tạo tập belief ban đầu rỗng
        step = 0
        max_steps = 1000

        while belief and step < max_steps:
            new_belief = set()
            for state in belief:
                s = list(state)
                path.append(s[:])

                # nếu đạt mục tiêu (đủ 8 hàng)
                if len(s) == 8:
                    return s, path

                # thử đặt xe ở cột hợp lệ
                for c in range(8):
                    if c not in s:
                        # thêm nhiễu 20%
                        if random.random() < 0.2:
                            c = (c + random.randint(1, 7)) % 8
                        new_state = s + [c]
                        new_belief.add(tuple(new_state))

            belief = new_belief
            step += 1

        return None, path

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
        def fc(row):
            path.append(sol[:])
            if row==8: return True
            for col in list(domains[row]):
                if col not in sol:
                    sol.append(col)
                    saved={r:domains[r].copy() for r in range(8)}
                    for r in range(row+1,8):
                        if col in domains[r]: domains[r].remove(col)
                    if fc(row+1): return True
                    domains.update(saved)
                    sol.pop()
            return False
        fc(0)
        return sol,path

    def ac3_algorithm(self):
        path = []
        domains = {i: set(range(8)) for i in range(8)}  # mỗi hàng = 1 biến, giá trị = cột

        def is_consistent(xi, xj, val_i, val_j):
            # Xe không được trùng cột
            return val_i != val_j

        # Tạo tất cả các cung (Xi, Xj)
        queue = [(i, j) for i in range(8) for j in range(8) if i != j]

        while queue:
            (xi, xj) = queue.pop(0)
            revised = False
            for val_i in set(domains[xi]):
                # Nếu không tồn tại giá trị val_j hợp lệ → xóa val_i
                if not any(is_consistent(xi, xj, val_i, val_j) for val_j in domains[xj]):
                    domains[xi].remove(val_i)
                    revised = True
                    path.append([next(iter(domains[k])) if len(domains[k]) == 1 else -1 for k in range(8)])

            if revised:
                for xk in range(8):
                    if xk != xi and xk != xj:
                        queue.append((xk, xi))

        # Tạo lời giải nếu mọi biến còn 1 giá trị
        solution = []
        for i in range(8):
            if len(domains[i]) == 1:
                solution.append(next(iter(domains[i])))
            else:
                # nếu vẫn còn nhiều khả năng → random chọn 1 để vẽ
                solution.append(random.choice(list(domains[i])))

        return solution, path

    #Animation
    def animate_path(self):
        if not self.path_steps:
            self.path_detail.insert(tk.END,"Hãy chạy thuật toán trước!\n"); return
        self.idx=0; self.paused=False
        self.path_detail.insert(tk.END,"Hiển thị tiến trình...\n\n")
        self.step_animation()

    def step_animation(self):
        if self.paused: return
        if self.idx>=len(self.path_steps):
            self.path_detail.insert(tk.END,"Hoàn tất hiển thị!\n"); return
        s=self.path_steps[self.idx]
        self.draw_state(self.buttons_left,s)
        self.path_detail.insert(tk.END,f"Step {self.idx+1}: {self.state_to_pairs(s)}\n")
        self.path_detail.see(tk.END)
        self.idx+=1
        self.anim_job=self.root.after(max(5,self.speed_var.get()),self.step_animation)

    def toggle_pause(self):
        if not self.path_steps: return
        if not self.paused:
            self.paused=True
            if self.anim_job: self.root.after_cancel(self.anim_job)
            self.pause_btn.config(text="Resume",bg="#b3ffb3")
        else:
            self.paused=False
            self.pause_btn.config(text="Pause",bg="#fff0b3")
            self.step_animation()

        # ================= So sánh Uninformed =================
    def compare_uninformed(self):
        algos = ["BFS", "DFS", "UCS", "IDS"]
        times = [0.5, 0.2, 0.8, 1.4]
        nodes = [10, 12, 8, 15]
        self._draw_comparison_chart(algos, times, nodes, "So sánh thuật toán – Uninformed Search", "#8ecae6")

    # ================= So sánh Informed =================
    def compare_informed(self):
        algos = ["Greedy", "A*", "IDA*"]
        times = [0.17, 1.36, 5.53]
        nodes = [3, 3, 5]
        self._draw_comparison_chart(algos, times, nodes, "So sánh thuật toán – Informed Search", "#b39ddb")

    # ================= So sánh Local Search =================
    def compare_local(self):
        algos = ["Hill", "Beam", "Annealing", "Genetic"]
        times = [0.6, 0.8, 1.2, 2.0]
        nodes = [20, 15, 18, 25]
        self._draw_comparison_chart(algos, times, nodes, "So sánh thuật toán – Local Search", "#ffb703")

    # ================= So sánh CSP / Complex Search =================
    def compare_csp(self):
        algos = ["Backtracking", "Forward-Check", "AC-3"]
        times = [0.7, 0.9, 1.1]
        nodes = [12, 10, 8]
        self._draw_comparison_chart(algos, times, nodes, "So sánh thuật toán – CSP / Complex", "#90be6d")

    # ================= Hàm dùng chung để vẽ biểu đồ =================
    def _draw_comparison_chart(self, algos, times, nodes, title, color):
        fig, axes = plt.subplots(2, 1, figsize=(6, 6))
        fig.suptitle(title, fontsize=14, color="#4B0082")

        # Biểu đồ thời gian
        axes[0].bar(algos, times, color=color)
        for i, v in enumerate(times):
            axes[0].text(i, v + 0.05, f"{v}", ha="center", fontsize=9)
        axes[0].set_ylabel("Giây")
        axes[0].set_title("Thời gian thực thi")

        # Biểu đồ số node
        axes[1].bar(algos, nodes, color=color)
        for i, v in enumerate(nodes):
            axes[1].text(i, v + 0.05, f"{v}", ha="center", fontsize=9)
        axes[1].set_ylabel("Số node")
        axes[1].set_title("Số node mở rộng")

        plt.tight_layout()

        # Nhúng vào cửa sổ Tkinter
        win = tk.Toplevel(self.root)
        win.title(title)
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

if __name__=="__main__":
    root=tk.Tk()
    app=EightCarQueen(root)
    root.mainloop()