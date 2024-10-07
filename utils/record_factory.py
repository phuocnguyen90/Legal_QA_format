# utils/record_factory.py

class RecordFactory:
    def __init__(self, llm_formatter):
        self.llm_formatter = llm_formatter

    def create_record_from_unformatted_text(self, text, record_type="DOC"):
        # Use LLMFormatter to format the text first
        formatted_text = self.llm_formatter.format_text(text, mode="tagged", record_type=record_type)
        if not formatted_text:
            raise ValueError("Failed to format text using LLMFormatter")

        # Now create a Record from the tagged text
        return Record.from_tagged_text(formatted_text, record_type=record_type)
