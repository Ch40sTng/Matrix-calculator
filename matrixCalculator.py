import tkinter as tk
from tkinter import ttk, messagebox
from scipy.linalg import lu
import numpy as np

class MatrixCalculator:
    def __init__(self, root):
        self.root = root
        self.root.title("Matrix Calculator")
        self.root.geometry("600x700")

        style = ttk.Style()
        style.configure("TLabel", font=("Arial", 11))
        style.configure("TButton", font=("Arial", 11))
        style.configure("TEntry", font=("Arial", 11))
        
        size_frame = ttk.Frame(root)
        size_frame.pack(pady=10)

        # Setting A size
        ttk.Label(size_frame, text="Matrix A (row x col)： ").grid(row=0, column=0)
        self.A_rows_var = tk.IntVar(value=2)
        self.A_cols_var = tk.IntVar(value=2)

        # input
        self.A_row_entry = ttk.Entry(size_frame, width=8, font=("Arial", 11))
        self.A_row_entry.grid(row=0, column=1)

        ttk.Label(size_frame, text=" x ").grid(row=0, column=2)

        self.A_col_entry = ttk.Entry(size_frame, width=8, font=("Arial", 11))
        self.A_col_entry.grid(row=0, column=3)

        self.A_row_entry.bind("<FocusIn>", self.clear_placeholder)
        self.A_col_entry.bind("<FocusIn>", self.clear_placeholder)
        self.A_row_entry.bind("<FocusOut>", lambda e: self.set_placeholder(self.A_row_entry, "A_row"))
        self.A_col_entry.bind("<FocusOut>", lambda e: self.set_placeholder(self.A_col_entry, "A_col"))

        # Setting B size
        ttk.Label(size_frame, text="Matrix B (row x col)： ").grid(row=1, column=0)
        self.B_rows_var = tk.IntVar(value=2)
        self.B_cols_var = tk.IntVar(value=2)

        # input
        self.B_row_entry = ttk.Entry(size_frame, width=8, font=("Arial", 11))
        self.B_row_entry.grid(row=1, column=1)

        ttk.Label(size_frame, text=" x ").grid(row=1, column=2)

        self.B_col_entry = ttk.Entry(size_frame, width=8, font=("Arial", 11))
        self.B_col_entry.grid(row=1, column=3)

        self.B_row_entry.bind("<FocusIn>", self.clear_placeholder)
        self.B_col_entry.bind("<FocusIn>", self.clear_placeholder)
        self.B_row_entry.bind("<FocusOut>", lambda e: self.set_placeholder(self.B_row_entry, "B_row"))
        self.B_col_entry.bind("<FocusOut>", lambda e: self.set_placeholder(self.B_col_entry, "B_col"))

        # confirm A
        self.confirm_button = ttk.Button(size_frame, text="Confirm A", command=self.update_matrixA, width=10)
        self.confirm_button.grid(row=0, column=4, padx=5)
        self.set_placeholder(self.A_row_entry, "A_row")
        self.set_placeholder(self.A_col_entry, "A_col")

        # confirm B
        self.confirm_button = ttk.Button(size_frame, text="Confirm B", command=self.update_matrixB, width=10)
        self.confirm_button.grid(row=1, column=4, padx=5)
        self.set_placeholder(self.B_row_entry, "B_row")
        self.set_placeholder(self.B_col_entry, "B_col")

        ttk.Label(size_frame, text="(Maximum size = 5 x 5)").grid(row=2, column=0, columnspan=4)

        # swap
        ttk.Button(size_frame, text="Swap A ⇄ B", command=self.swap).grid(row=0, column=5, rowspan=2, padx=10)

        # input
        self.matrixA_frame = ttk.Frame(root)
        self.matrixA_frame.pack(pady=5)
        self.matrixB_frame = ttk.Frame(root)
        self.matrixB_frame.pack(pady=5)

        self.create_matrixA_entries()
        self.create_matrixB_entries()

        # buttom
        btn_frame = ttk.Frame(root)
        btn_frame.pack(pady=10)

        ttk.Button(btn_frame, text="A + B", command=self.add).grid(row=0, column=0, padx=5)
        ttk.Button(btn_frame, text="A - B", command=self.sub).grid(row=0, column=1, padx=5)
        ttk.Button(btn_frame, text="A x B", command=self.mul).grid(row=0, column=2, padx=5)

        ttk.Button(btn_frame, text="Inverse", command=self.inverse).grid(row=1, column=0, padx=5)
        ttk.Button(btn_frame, text="Determinant", command=self.det).grid(row=1, column=1, padx=5)
        ttk.Button(btn_frame, text="Transpose", command=self.trans).grid(row=1, column=2, padx=5)
        ttk.Button(btn_frame, text="Rank", command=self.rank).grid(row=1, column=3, padx=5)        

        ttk.Button(btn_frame, text="REF", command=self.row_echelon).grid(row=2, column=0, padx=5)
        ttk.Button(btn_frame, text="Diagonal", command=self.diagonal).grid(row=2, column=1, padx=5)
        ttk.Button(btn_frame, text="LU", command=self.lu_decomposition).grid(row=3, column=0, padx=5)
        ttk.Button(btn_frame, text="Eigen", command=self.eigen).grid(row=3, column=1, padx=5)

        # Multiply by factor
        ttk.Button(btn_frame, text="Multiply by", command=self.mul_factor).grid(row=2, column=2)
        self.factor_entry = ttk.Entry(btn_frame, width=5, foreground="gray")
        self.factor_entry.grid(row=2, column=3,sticky="w")
        self.factor_entry.insert(0, "MUL")
        self.factor_entry.bind("<FocusIn>", self.clear_placeholder)
        self.factor_entry.bind("<FocusOut>", lambda e: self.set_placeholder(self.factor_entry, "MUL"))

        
        # Power function
        ttk.Button(btn_frame, text="The Power of", command=self.power).grid(row=3, column=2)
        self.power_entry = ttk.Entry(btn_frame, width=5, foreground="gray")
        self.power_entry.grid(row=3, column=3,sticky="w")
        self.power_entry.insert(0, "EXP")
        self.power_entry.bind("<FocusIn>", self.clear_placeholder)
        self.power_entry.bind("<FocusOut>", lambda e: self.set_placeholder(self.power_entry, "EXP"))

         # result
        self.result_label = ttk.Label(root, text="Result:", font=("Arial", 12, "bold"))
        self.result_label.pack()
        self.result_text = tk.Text(root, height=5, width=40, font=("Arial", 11))
        self.result_text.pack(pady=5)
        self.result_text.config(state="disabled")

        # Store Buttons
        store_frame = ttk.Frame(root)
        store_frame.pack()
        self.store_A_btn = ttk.Button(store_frame, text="Store to A", command=self.store_to_A, state="disabled")
        self.store_A_btn.grid(row=0, column=0, pady=10)

        self.store_B_btn = ttk.Button(store_frame, text="Store to B", command=self.store_to_B, state="disabled")
        self.store_B_btn.grid(row=0, column=1, pady=10)

# ------------------------------------------------function---------------------------------------------------------------
    def set_placeholder(self, entry, text):
        if not entry.get():
            entry.insert(0, text)
            entry.config(foreground="gray")

    def clear_placeholder(self, event):
        if event.widget.get().startswith(("A", "B", "E", "F", "M")):
            event.widget.delete(0, tk.END)
            event.widget.config(foreground="black")

    def update_matrixA(self):
        try:
            rows_a = int(self.A_row_entry.get())
            cols_a = int(self.A_col_entry.get())

            if all(1 <= x <= 5 for x in [rows_a, cols_a]):
                self.A_rows_var.set(rows_a)
                self.A_cols_var.set(cols_a)
                self.create_matrixA_entries()
            else:
                messagebox.showerror("Error", "Rows and Columns must be between 1 and 5")
        except ValueError:
            messagebox.showerror("Error", "Please enter valid integers")

    def update_matrixB(self):
        try:
            rows_b = int(self.B_row_entry.get())
            cols_b = int(self.B_col_entry.get())

            if all(1 <= x <= 5 for x in [rows_b, cols_b]):
                self.B_rows_var.set(rows_b)
                self.B_cols_var.set(cols_b)
                self.create_matrixB_entries()
            else:
                messagebox.showerror("Error", "Rows and Columns must be between 1 and 5")
        except ValueError:
            messagebox.showerror("Error", "Please enter valid integers")

    def create_matrixA_entries(self):
        for widget in self.matrixA_frame.winfo_children():
            widget.destroy()

        A_rows, A_cols = self.A_rows_var.get(), self.A_cols_var.get()

        self.entries1 = []

        ttk.Label(self.matrixA_frame, text="Matrix A: ").grid(row=0, column=0, columnspan=A_cols)
        for i in range(A_rows):
            row_entries = []
            for j in range(A_cols):
                entry = ttk.Entry(self.matrixA_frame, width=6, font=("Arial", 11), foreground="gray")
                entry.grid(row=i+1, column=j, padx=2, pady=2)

                entry.insert(0, f"A{i+1}{j+1}")
                entry.bind("<FocusIn>", self.clear_placeholder)
                entry.bind("<FocusOut>", lambda e: self.set_placeholder(entry, f"A{i+1}{j+1}"))

                row_entries.append(entry)
            self.entries1.append(row_entries)

        ttk.Label(self.matrixA_frame, text="").grid(row=A_rows + 1, column=0)

    def create_matrixB_entries(self):
        for widget in self.matrixB_frame.winfo_children():
            widget.destroy()

        B_rows, B_cols = self.B_rows_var.get(), self.B_cols_var.get()

        self.entries2 = []

        ttk.Label(self.matrixB_frame, text="Matrix B: ").grid(row=0, column=0, columnspan=B_cols)
        for i in range(B_rows):
            row_entries = []
            for j in range(B_cols):
                entry = ttk.Entry(self.matrixB_frame, width=6, font=("Arial", 11), foreground="gray")
                entry.grid(row=i+1, column=j, padx=2, pady=2)

                entry.insert(0, f"B{i+1}{j+1}")
                entry.bind("<FocusIn>", self.clear_placeholder)
                entry.bind("<FocusOut>", lambda e: self.set_placeholder(entry, f"B{i+1}{j+1}"))

                row_entries.append(entry)
            self.entries2.append(row_entries)

    #get value from input
    def get_matrix(self, entries):
        try:
            return np.array([[float(entry.get()) for entry in row] for row in entries])
        except ValueError:
            messagebox.showerror("Error", "Invalid Input")
            return None

    #swap function
    def swap(self):
        # get data
        data_A = [[entry.get().strip() for entry in row] for row in self.entries1]
        data_B = [[entry.get().strip() for entry in row] for row in self.entries2]

        # get size
        rows_A, cols_A = self.A_rows_var.get(), self.A_cols_var.get()
        rows_B, cols_B = self.B_rows_var.get(), self.B_cols_var.get()

        # switch size
        self.A_rows_var.set(rows_B)
        self.A_cols_var.set(cols_B)
        self.B_rows_var.set(rows_A)
        self.B_cols_var.set(cols_A)

        self.create_matrixA_entries()
        self.create_matrixB_entries()

        for i in range(min(rows_B, len(self.entries1))):
            for j in range(min(cols_B, len(self.entries1[i]))):
                value = data_B[i][j]
                if value and not value.startswith("B"):
                    self.entries1[i][j].delete(0, tk.END)
                    self.entries1[i][j].insert(0, value)
                    self.entries1[i][j].config(foreground="black")

        for i in range(min(rows_A, len(self.entries2))):
            for j in range(min(cols_A, len(self.entries2[i]))):
                value = data_A[i][j]
                if value and not value.startswith("A"):
                    self.entries2[i][j].delete(0, tk.END)
                    self.entries2[i][j].insert(0, value)
                    self.entries2[i][j].config(foreground="black") 

        self.update_placeholder(self.entries1, "A")
        self.update_placeholder(self.entries2, "B")


    def update_placeholder(self, entries, prefix):
        for i, row in enumerate(entries):
            for j, entry in enumerate(row):
                if not entry.get():
                    self.set_placeholder(entry, f"{prefix}{i+1}{j+1}")


    #result function
    def store_result(self, target):
        if self.result_matrix is None:
            return
        
        rows, cols = self.result_matrix.shape
        if target == "A":
            self.A_rows_var.set(rows)
            self.A_cols_var.set(cols)
            self.create_matrixA_entries()
            target_entries = self.entries1
        else:
            self.B_rows_var.set(rows)
            self.B_cols_var.set(cols)
            self.create_matrixB_entries()
            target_entries = self.entries2

        for i in range(rows):
            for j in range(cols):
                target_entries[i][j].delete(0, tk.END)
                target_entries[i][j].insert(0, str(self.result_matrix[i, j]))
                target_entries[i][j].config(foreground="black")

    def store_to_A(self):
        self.store_result("A")

    def store_to_B(self):
        self.store_result("B")

    def show_result(self, result, store=True):
        self.result_text.config(state="normal")
        self.result_text.delete("1.0", tk.END)

        if isinstance(result, str):
            result_text = result
            self.result_matrix = None
        else:
            self.result_matrix = result
            result_text = "\n".join(["  ".join(f"{val:.2f}" for val in row) for row in result])

        self.result_text.insert(tk.END, result_text)
        self.result_text.config(state="disabled")

        if store and self.result_matrix is not None:
            self.store_A_btn.config(state="normal")
            self.store_B_btn.config(state="normal")
        else:
            self.store_A_btn.config(state="disabled")
            self.store_B_btn.config(state="disabled")
         
    #calulate function
    def add(self):
        A, B = self.get_matrix(self.entries1), self.get_matrix(self.entries2)
        if A is not None and B is not None:
            try:
                result = A + B
                self.show_result(result)
            except ValueError:
                messagebox.showerror("Error", "Dimension Error")

    def sub(self):
        A, B = self.get_matrix(self.entries1), self.get_matrix(self.entries2)
        if A is not None and B is not None:
            try:
                result = A - B
                self.show_result(result)
            except ValueError:
                messagebox.showerror("Error", "Dimension Error")

    def mul(self):
        A, B = self.get_matrix(self.entries1), self.get_matrix(self.entries2)
        if A is not None and B is not None:
            try:
                result = np.dot(A, B)
                self.show_result(result)
            except ValueError:
                messagebox.showerror("Error", "Dimension Error")

    def inverse(self):
        A = self.get_matrix(self.entries1)
        if A is None:
            return
        if A.shape[0] != A.shape[1]:
            messagebox.showerror("Error", "This matrix must be square.")
            return
        try:
            result = np.linalg.inv(A)
            self.show_result(result)
        except np.linalg.LinAlgError:
            messagebox.showerror("Error", "This matrix is not invertible.")
    
    def det(self):
        A = self.get_matrix(self.entries1)
        if A is None:
            return

        if A.shape[0] != A.shape[1]:  
            messagebox.showerror("Error", "This matrix must be square.")
            return

        try:
            determinant = np.linalg.det(A)
            self.show_result(f"{determinant:.2f}", store=False)
        except np.linalg.LinAlgError:
            messagebox.showerror("Error", "Failed to compute determinant.")

    def trans(self):
        A = self.get_matrix(self.entries1)
        if A is not None:
            self.show_result(A.T)

    def rank(self):
        A = self.get_matrix(self.entries1)
        if A is not None:
            self.show_result(np.linalg.matrix_rank(A), store=False)
    
    def mul_factor(self):
        A = self.get_matrix(self.entries1)
        try:
            factor = float(self.factor_entry.get())
            self.show_result(A * factor)
        except ValueError:
            messagebox.showerror("Error", "Invalid Factor")

    def power(self):
        A = self.get_matrix(self.entries1)
        if A.shape[0] != A.shape[1]:
            messagebox.showerror("Error", "Matrix must be square for power operation")
            return
        try:
            power = int(self.power_entry.get())
            self.show_result(np.linalg.matrix_power(A, power))
        except ValueError:
            messagebox.showerror("Error", "Invalid Power")
        except np.linalg.LinAlgError:
            messagebox.showerror("Error", "Matrix power calculation failed")
    
    def row_echelon(self):
        A = self.get_matrix(self.entries1)

        if A is not None:
            A = A.astype(float)  # 轉換為浮點數，避免整數運算問題
            rows, cols = A.shape
            pivot_row = 0

            for col in range(cols):
                if pivot_row >= rows:
                    break
                
                nonzero_rows = np.where(A[pivot_row:, col] != 0)[0]
                if len(nonzero_rows) == 0:
                    continue

                first_nonzero_row = pivot_row + nonzero_rows[0]
                A[[pivot_row, first_nonzero_row]] = A[[first_nonzero_row, pivot_row]]

                for r in range(pivot_row + 1, rows):
                    factor = A[r, col] / A[pivot_row, col]
                    A[r] -= factor * A[pivot_row]

                pivot_row += 1

        A = np.where(np.abs(A) < 1e-10, 0, A)
        self.show_result(A)

    def diagonal(self):
        A = self.get_matrix(self.entries1)
        if A.shape[0] != A.shape[1]:  
            messagebox.showerror("Error", "This matrix must be square.")
            return
    
        if A is not None:
            try:
                # Compute the eigenvalues and eigenvectors
                eigenvalues, eigenvectors = np.linalg.eig(A)

                # Create the diagonal matrix D (containing eigenvalues)
                D = np.diag(eigenvalues)
                P = eigenvectors
                P_inv = np.linalg.inv(P)

                result_text = "P inverse:\n"
                result_text += "\n".join(["  ".join(f"{val:.2f}" for val in row) for row in P_inv]) + "\n\n"

                result_text += "D (Diagnal Matrix):\n"
                result_text += "\n".join(["  ".join(f"{val:.2f}" for val in row) for row in D]) + "\n\n"

                result_text += "P (Eigenvector Matrix):\n"
                result_text += "\n".join(["  ".join(f"{val:.2f}" for val in row) for row in P])

                self.show_result(result_text, store=False)

            except Exception as e:
                messagebox.showerror("Error", f"Diagonalization failed: {str(e)}")
    
    def lu_decomposition(self):
        A = self.get_matrix(self.entries1)
        if A is not None:
            try:
                P, L, U = lu(A)
                result_text = "P (Permutation Matrix):\n"
                result_text += "\n".join(["  ".join(f"{val:.2f}" for val in row) for row in P]) + "\n\n"

                result_text += "L (Lower Triangular Matrix):\n"
                result_text += "\n".join(["  ".join(f"{val:.2f}" for val in row) for row in L]) + "\n\n"

                result_text += "U (Upper Triangular Matrix):\n"
                result_text += "\n".join(["  ".join(f"{val:.2f}" for val in row) for row in U])

                self.show_result(result_text, store=False)
            except Exception as e:
                messagebox.showerror("Error", f"LU decomposition failed: {e}")
    
    def eigen(self):
        A = self.get_matrix(self.entries1)
        if A is not None and A.shape[0] == A.shape[1]:
            try:
                values, vectors = np.linalg.eig(A)
                result_text = "Eigenvalues:\n"
                result_text += "\n".join(f"λ{i+1} = {val:.4f}" for i, val in enumerate(values)) + "\n\n"

                result_text += "Eigenvectors (columns):\n"
                for i, vec in enumerate(vectors.T):
                    result_text += f"v{i+1} = [" + "  ".join(f"{val:.4f}" for val in vec) + "]\n"

                self.show_result(result_text, store=False) 
            except np.linalg.LinAlgError:
                messagebox.showerror("Error", "Eigen decomposition failed.")
        else:
            messagebox.showerror("Error", "Matrix must be square.")


root = tk.Tk()
app = MatrixCalculator(root)
root.mainloop()
