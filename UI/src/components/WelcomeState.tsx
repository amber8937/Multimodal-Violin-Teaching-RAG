import { ChatInput } from './ChatInput';

interface WelcomeStateProps {
  onSuggestionClick?: (suggestion: string) => void;
  onSend: (message: string) => void;
  isLoading?: boolean;
}

export function WelcomeState({ onSuggestionClick, onSend, isLoading = false }: WelcomeStateProps) {
  return (
    <div className="flex flex-col items-center justify-center h-full px-8 relative">
      {/* Main content centered with better vertical balance */}
      <div className="flex flex-col items-center" style={{ marginTop: '-12vh' }}>
        {/* Cat violin logo */}
        <div className="mb-6">
          <img
            src="/cat-violin.png"
            alt="Cat playing violin"
            width="150"
            height="150"
            className="object-contain"
          />
        </div>

        {/* App name */}
        <h1 className="text-[30px] font-semibold text-[#1e3a5f] mb-10 text-center">
          Violin Teaching Assistant
        </h1>

        {/* Search input - centered on welcome screen, longer width */}
        <div className="w-full max-w-[820px] mb-5">
          <ChatInput onSend={onSend} centered isLoading={isLoading} />
        </div>

        {/* Tagline */}
        <p className="text-[14px] text-[#888888] text-center">
          Find answers to your practice problems
        </p>
      </div>

      {/* AI disclaimer at bottom */}
      <div className="absolute bottom-6 left-0 right-0">
        <p className="text-[11px] text-[#999999] text-center">
          AI-generated answers may be inaccurate. Always verify with source documents.
        </p>
      </div>
    </div>
  );
}