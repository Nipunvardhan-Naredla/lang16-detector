from lang16_detector import query_model
import customtkinter as ctk

#Functions
def get_text():

    #prints loading text
    confidence.configure(state = "normal")
    language.configure(state = "normal")
    confidence.delete("1.0", ctk.END)
    language.delete("1.0", ctk.END)
    confidence.insert("0.0", " "*35 + "Confidence: Loading...")
    language.insert("0.0", " "*35 + "Language: Loading...")
    confidence.configure(state = "disabled")
    language.configure(state = "disabled")

    #querys model
    answer = query_model(input_box.get("0.0", "end-1c"))

    #prints outputs
    confidence.configure(state = "normal")
    language.configure(state = "normal")
    confidence.delete("1.0", ctk.END)
    language.delete("1.0", ctk.END)
    confidence.insert("0.0", " "*35 + "Confidence: " + str(answer["confidence"]) + "%")
    language.insert("0.0", " "*35 + "Language: " + str(answer["class"]))
    confidence.configure(state = "disabled")
    language.configure(state = "disabled")
    
    

#creates window
window = ctk.CTk()
window.geometry("640x480")
window.title("lang16-detector")
window.resizable(False, False)

#creates title
title = ctk.CTkTextbox(master=window,
                       width=640,
                       height=20,
                       corner_radius=0,
                       font = ctk.CTkFont(family="Helvetica", size=24, weight="bold"),
                       )
title.place(x=0, y=20)
title.insert("0.0", " "*35 + "lang16-detector")
title.configure(state = "disabled")

#creates input box
input_box = ctk.CTkTextbox(master=window, width=300, height=150)
input_box.place(x=170, y=100)
input_box.insert("0.0", "Cookies are better than cakes only if they don't have raisins")

#creates enter button
enter_button = ctk.CTkButton(master=window, text="Process", command=get_text)
enter_button.place(x=250, y = 280)

#creates output text
#Language
language = ctk.CTkTextbox(master=window,
                       width=640,
                       height=20,
                       corner_radius=0,
                       font = ctk.CTkFont(family="Helvetica", size=18, weight="bold"),
                       )
language.place(x=0, y=330)
language.insert("0.0", " "*35 + "Language:")
language.configure(state = "disabled")

#Confidence
confidence = ctk.CTkTextbox(master=window,
                       width=640,
                       height=20,
                       corner_radius=0,
                       font = ctk.CTkFont(family="Helvetica", size=18, weight="bold"),
                       )
confidence.place(x=0, y=370)
confidence.insert("0.0", " "*35 + "Confidence:")
confidence.configure(state = "disabled")

window.mainloop()

def query():
    pass
    #print(query_model("Cookies are better than cakes only if they don't have raisins"))
    
