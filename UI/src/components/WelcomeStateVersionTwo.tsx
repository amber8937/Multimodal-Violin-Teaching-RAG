import { useState } from 'react';
import { ChatInput } from './ChatInput';

interface WelcomeStateVersionTwoProps {
  onSuggestionClick?: (suggestion: string) => void;
  onSend: (message: string) => void;
  isLoading?: boolean;
}

const exampleQuestions = [
  'Expressive and rich sound',
  'Vibrato exercises',
  'Shifting smoothly',
  'Reduce left hand tension',
  'Bow hold technique'
];

export function WelcomeStateVersionTwo({
  onSuggestionClick,
  onSend,
  isLoading = false
}: WelcomeStateVersionTwoProps) {
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);

  const handleSuggestion = (suggestion: string) => {
    if (onSuggestionClick) {
      onSuggestionClick(suggestion);
      return;
    }
    onSend(suggestion);
  };

  return (
    <div className="flex flex-col items-center justify-center h-full px-8 relative">
      <div className="flex flex-col items-center w-full" style={{ marginTop: '-12vh' }}>
        <div className="mb-6">
          <img
            src="/cat_violin.png"
            alt="Cat playing violin"
            width="150"
            height="150"
            className="object-contain drop-shadow-sm"
          />
        </div>

        <h1 className="text-lg font-semibold text-[#1e3a5f] mb-6 text-center" style={{ fontFamily: "'EB Garamond', serif", fontSize: '32px', letterSpacing: '-0.5px', fontWeight: '600', lineHeight: '1.2' }}>
          Violin Teaching Assistant
        </h1>

        <div className="w-full max-w-[820px] mb-10">
          <ChatInput
            onSend={onSend}
            centered
            isLoading={isLoading}
            placeholder="Ask a question about violin playing..."
          />
        </div>

        {/* Example Questions */}
        <div className="w-full max-w-[800px]">
          <div className="flex flex-wrap justify-center gap-3">
            {exampleQuestions.map((question, index) => (
              <button
                key={question}
                type="button"
                onClick={() => handleSuggestion(question)}
                onMouseEnter={() => setHoveredIndex(index)}
                onMouseLeave={() => setHoveredIndex(null)}
                style={{
                  padding: '10px 20px',
                  borderRadius: '20px',
                  fontSize: '14px',
                  fontWeight: '500',
                  cursor: 'pointer',
                  border: hoveredIndex === index ? '1.5px solid #2f6da8' : '1.5px solid #e2e8ed',
                  backgroundColor: hoveredIndex === index ? '#f0f7ff' : '#ffffff',
                  color: hoveredIndex === index ? '#215f9a' : '#4a5d6a',
                  transition: 'all 0.25s ease-out',
                  boxShadow: hoveredIndex === index
                    ? '0 2px 10px rgba(74, 143, 200, 0.17)'
                    : '0 1px 3px rgba(0, 0, 0, 0.08)',
                  outline: 'none'
                }}
                onFocus={(e) => {
                  e.currentTarget.style.outline = 'none';
                  e.currentTarget.style.boxShadow = '0 0 0 4px rgba(74, 144, 200, 0.08)';
                }}
                onBlur={(e) => {
                  if (hoveredIndex === index) {
                    e.currentTarget.style.boxShadow = '0 2px 10px rgba(74, 144, 200, 0.12)';
                  } else {
                    e.currentTarget.style.boxShadow = '0 1px 3px rgba(0, 0, 0, 0.08)';
                  }
                }}
              >
                {question}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="absolute bottom-6 left-0 right-0">
        <p className="text-[11px] text-[#999999] text-center">
          AI-generated answers may be inaccurate. Always verify with source documents.
        </p>
      </div>
    </div>
  );
}
