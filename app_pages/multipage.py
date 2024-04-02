import streamlit as st


# Object-oriented function to generate Streamlit pages
class MultiPage:

    def __init__(self, app_name) -> None:
        self.pages = []
        self.app_name = app_name

        st.set_page_config(
            page_title=self.app_name,
            page_icon="🌱")

    def add_page(self, title, func) -> None:
        self.pages.append({"title": title, "function": func})

    def run(self):
        st.title(self.app_name)

        def format_page_title(page):
            return page['title']
        page = st.sidebar.radio('Menu', self.pages,
                                format_func=format_page_title)
        page['function']()
