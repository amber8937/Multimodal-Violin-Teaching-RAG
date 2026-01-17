import { useState } from 'react';
import { ArrowUp } from 'lucide-react';

interface ChatInputProps {
  onSend: (message: string) => void;
  centered?: boolean;
  isLoading?: boolean;
  placeholder?: string;
}

export function ChatInput({ onSend, centered = false, isLoading = false, placeholder = "Ask a question..." }: ChatInputProps) {
  const [input, setInput] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (input.trim()) {
      onSend(input.trim());
      setInput('');
    }
  };

  return (
    <div className={centered ? "" : "bg-[#fafafa] pb-6 px-6"}>
      <div className={centered ? "w-full" : "max-w-3xl mx-auto"}>
        <form onSubmit={handleSubmit} className="relative">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder={placeholder}
            disabled={isLoading}
            style={{
              border: '1.5px solid #d0d0d0',
            }}
            className={`w-full px-4 pr-14 rounded-xl bg-white shadow-[0_2px_8px_rgba(0,0,0,0.06)] focus:outline-none placeholder:text-[#999999] text-[15px] transition-all duration-200 ${
              centered ? 'h-[52px]' : 'h-12'
            } ${isLoading ? 'opacity-60 cursor-not-allowed' : ''}`}
            onFocus={(e) => {
              e.currentTarget.style.border = '1.5px solid #2f6da8';
              e.currentTarget.style.boxShadow = '0 0 0 4px rgba(74, 144, 200, 0.08)';
            }}
            onBlur={(e) => {
              e.currentTarget.style.border = '1.5px solid #d0d0d0';
              e.currentTarget.style.boxShadow = '0 2px 8px rgba(0,0,0,0.06)';
            }}
          />
          <button
            type="submit"
            disabled={!input.trim() || isLoading}
            className={`absolute right-2 top-1/2 -translate-y-1/2 rounded-lg flex items-center justify-center transition-colors ${
              centered ? 'w-9 h-9' : 'w-8 h-8'
            } ${
              input.trim() && !isLoading
                ? 'bg-[#2563eb] hover:bg-[#3b82f6]'
                : 'bg-[#e0e0e0]'
            }`}
            aria-label="Send message"
          >
            <ArrowUp className="w-5 h-5 text-white" />
          </button>
        </form>
        {!centered && (
          <div className="text-[11px] text-[#999999] text-center mt-3">
            AI-generated answers may be inaccurate. Always verify with source documents.
          </div>
        )}
      </div>
    </div>
  );
}