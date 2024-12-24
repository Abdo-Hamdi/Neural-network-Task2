import tkinter as tk
from tkinter import messagebox, ttk
from Pre import Preprocess, Backprop, Test


class Gui(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Task 2")
        self.geometry("1200x900")
        self.container = ttk.Frame(self)
        self.container.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.input_parameter = ttk.Frame(self.container)
        self.output_parameter = ttk.Frame(self.container)
        self.learning_rate = None
        self.neuron_testbox = None
        self.layer_textbox = None
        self.num_of_epoch = None
        self.selected_activation_function = tk.IntVar()
        self.bias_var = tk.BooleanVar()
        self.matrix_entries = None
        self.overall_entry = None
        self.overall_entry_train = None
        self.input_parameter.place(relx=0, rely=0, relwidth=1, relheight=0.40)
        self.output_parameter.place(relx=0, rely=0.5, relwidth=1, relheight=0.60)
        self.create_widgets()
        self.matrix_entries = self.create_matrix_entries(self.output_parameter)
        self.mainloop()

    def create_widgets(self):
        # feature
        (tk.Label(self.input_parameter, text="Number of Hidden Layer:", font=('JetBrains Mono', 15)).grid(row=1,
                                                                                                          column=0,
                                                                                                          sticky='e',
                                                                                                          padx=10,
                                                                                                          pady=5))
        self.layer_textbox = tk.Entry(self.input_parameter, width=20, font=('JetBrains Mono', 15))
        self.layer_textbox.grid(row=1, column=1, padx=10, pady=5, sticky="w")
        # class
        tk.Label(self.input_parameter, text="Number of Neurons in each Layer:", font=('JetBrains Mono', 15)).grid(row=2,
                                                                                                                  column=0,
                                                                                                                  sticky='e',
                                                                                                                  padx=10,
                                                                                                                  pady=5)
        self.neuron_testbox = tk.Entry(self.input_parameter, width=20, font=('JetBrains Mono', 15))
        self.neuron_testbox.grid(row=2, column=1, padx=10, pady=5, sticky="w")

        tk.Label(self.input_parameter, text="Select Activation Function:", font=('JetBrains Mono', 15)).grid(row=3,
                                                                                                             column=0,
                                                                                                             sticky='e',
                                                                                                             padx=5,
                                                                                                             pady=5)
        tk.Radiobutton(self.input_parameter, text="Sigmoid", variable=self.selected_activation_function, value=1,
                       font=('JetBrains Mono', 15)).grid(row=3, column=1, sticky='w')
        tk.Radiobutton(self.input_parameter, text="Tanh", variable=self.selected_activation_function, value=2,
                       font=('JetBrains Mono', 15)).grid(
            row=3, column=2, sticky='w')
        # epoch
        tk.Label(self.input_parameter, text="Number of Epochs (m):", font=('JetBrains Mono', 15)).grid(row=4, column=0,
                                                                                                       sticky='e',
                                                                                                       padx=5,
                                                                                                       pady=5)
        self.num_of_epoch = tk.Entry(self.input_parameter, width=20, font=('JetBrains Mono', 15))
        self.num_of_epoch.grid(row=4, column=1, padx=5, pady=5)
        # learning rate
        tk.Label(self.input_parameter, text="Learning Rate (eta):", font=('JetBrains Mono', 15)).grid(row=5, column=0,
                                                                                                      sticky='e',
                                                                                                      padx=5, pady=5)
        self.learning_rate = tk.Entry(self.input_parameter, width=20, font=('JetBrains Mono', 15))
        self.learning_rate.grid(row=5, column=1, padx=5, pady=5)
        # Add Bias
        tk.Label(self.input_parameter, text="Add Bias:", font=('JetBrains Mono', 15)).grid(row=6, column=0, sticky='e',
                                                                                           padx=5,
                                                                                           pady=5)
        tk.Checkbutton(self.input_parameter, variable=self.bias_var, font=('JetBrains Mono', 15)).grid(row=6, column=1,
                                                                                                       pady=10)
        # run & exit buttons
        tk.Button(self.input_parameter, text="Run", command=self.run, padx=25, pady=10, background='blue',
                  font=('JetBrains Mono', 10)).grid(row=7, column=0, padx=10, pady=5)

        tk.Button(self.input_parameter, text="Exit", command=self.quit, padx=25, pady=10, background='red',
                  font=('JetBrains Mono', 10)).grid(row=7, column=1, padx=10, pady=5)

        tk.Label(self.output_parameter, text="Output", font=('JetBrains Mono', 20, 'bold')).grid(row=0, column=0,
                                                                                                 padx=10,
                                                                                                 pady=5)
        tk.Label(self.output_parameter, text="Overall Test Accuracy:", font=('JetBrains Mono', 15, 'bold')).grid(row=7,
                                                                                                            column=0,
                                                                                                            padx=10,
                                                                                                            pady=5)
        self.overall_entry = tk.Entry(self.output_parameter, width=20, font=('JetBrains Mono', 15))
        self.overall_entry.grid(row=7, column=1, padx=5, pady=5)
        tk.Label(self.output_parameter, text="Overall Train Accuracy:", font=('JetBrains Mono', 15, 'bold')).grid(row=6,
                                                                                                            column=0,
                                                                                                            padx=10,
                                                                                                            pady=5)
        self.overall_entry_train = tk.Entry(self.output_parameter, width=20, font=('JetBrains Mono', 15))
        self.overall_entry_train.grid(row=6, column=1, padx=5, pady=5)

        tk.Label(self.output_parameter, text="Predicted", font=('JetBrains Mono', 15, 'bold')).grid(row=1, column=1,
                                                                                                    columnspan=2,
                                                                                                    padx=10,
                                                                                                    pady=5)
        tk.Label(self.output_parameter, text="Actual", font=('JetBrains Mono', 15, 'bold')).grid(row=1, column=0,
                                                                                                 rowspan=2,
                                                                                                 padx=10, pady=5)

        tk.Label(self.output_parameter, text="Class 0", font=('JetBrains Mono', 12)).grid(row=2, column=1, padx=5, pady=5)
        tk.Label(self.output_parameter, text="Class 1", font=('JetBrains Mono', 12)).grid(row=2, column=2, padx=5, pady=5)
        tk.Label(self.output_parameter, text="Class 2", font=('JetBrains Mono', 12)).grid(row=2, column=3, padx=5, pady=5)

        tk.Label(self.output_parameter, text="Class 0", font=('JetBrains Mono', 12)).grid(row=3, column=0, padx=5, pady=5)
        tk.Label(self.output_parameter, text="Class 1", font=('JetBrains Mono', 12)).grid(row=4, column=0, padx=5, pady=5)
        tk.Label(self.output_parameter, text="Class 2", font=('JetBrains Mono', 12)).grid(row=5, column=0, padx=5, pady=5)

    def run(self):
        print("I'm here in run function")
        print("number of layer:", self.layer_textbox.get())
        print("number of neuron:", self.neuron_testbox.get())
        print("Selected activation:", self.selected_activation_function.get())
        print("Number of epochs:", self.num_of_epoch.get())
        print("Learning rate:", self.learning_rate.get())
        print("Add Bias:", self.bias_var.get())

        if (not self.layer_textbox.get() or not self.neuron_testbox.get()
                or self.selected_activation_function.get() == 0
                or not self.num_of_epoch.get() or not self.learning_rate.get()):
            messagebox.showerror("Input error", "You must enter all required values")
            return

        number_of_layer = self.layer_textbox.get()
        number_of_neu = self.neuron_testbox.get()
        epoch = int(self.num_of_epoch.get())
        eta = float(self.learning_rate.get())
        add_bias = bool(self.bias_var.get())
        activation_function = int(self.selected_activation_function.get())

        pre_app = Preprocess()
        x_train, x_test, y_train, y_test = pre_app.preprocessing()

        model = Backprop(epoch, add_bias, eta, number_of_layer, number_of_neu, activation_function)
        model.Train(x_train, y_train)

        test_phase = Test(x_test, y_test, model.weights, model.bias, activation_function, number_of_layer)
        confusion_matrix, accuracy = test_phase.test()
        train_accuracy = test_phase.get_acc(x_train, y_train)
        self.overall_entry.delete(0, 'end')
        self.overall_entry.insert(0, str(f"{accuracy * 100 : 0.2f}%"))
        self.overall_entry_train.delete(0, 'end')
        self.overall_entry_train.insert(0, str(f"{train_accuracy * 100 : 0.2f}%"))
        self.set_matrix_values(confusion_matrix)

    def create_matrix_entries(self, parent):
        labels = [
            ['TP-0', 'FP-0', 'FP-1'],
            ['FN-0', 'TP-1', 'FP-2'],
            ['FN-1', 'FN-2', 'TP-2']
        ]
        matrix_entries = []
        for i in range(3):
            row_entries = []
            for j in range(3):
                cell_frame = tk.Frame(parent)
                cell_frame.grid(row=i + 3, column=j + 1, padx=5, pady=5)
                tk.Label(cell_frame, text=labels[i][j], font=('JetBrains Mono', 10)).pack(anchor='n')
                entry = tk.Entry(cell_frame, width=15, font=('JetBrains Mono', 15), justify='center', state='readonly')
                entry.pack(anchor='s')
                row_entries.append(entry)
            matrix_entries.append(row_entries)
        return matrix_entries

    def set_matrix_values(self, confusion_matrix):
        values = confusion_matrix.tolist()
        for i in range(3):
            for j in range(3):
                var = tk.StringVar()
                self.matrix_entries[i][j].config(textvariable=var)
                var.set(str(values[i][j]))


if __name__ == '__main__':
    gui_app = Gui()
