from open_cortex.ui.app import build_app

demo = build_app()

if __name__ == "__main__":
    demo.queue()
    demo.launch()