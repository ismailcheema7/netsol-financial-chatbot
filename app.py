import gradio as gr
from functions import create_interface, running_agent  # make sure running_agent is also imported

if __name__ == "__main__":
    print("NetSol Financial Chatbot initialized!")
    print("Choose an option:")
    print("1. Launch Gradio interface")
    print("2. Run command line interface")

    choice = input("Enter your choice (1 or 2): ")

    if choice == "1":
        demo = create_interface()
        demo.launch(
            share=True,
            server_name="0.0.0.0",
            server_port=7860,
            debug=True
        )
    elif choice == "2":
        running_agent()
    else:
        print("❌ Invalid choice. Please run the script again and choose 1 or 2.")
