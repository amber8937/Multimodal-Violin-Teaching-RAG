import type { Message } from '../App';

interface UserMessageProps {
  message: Message;
}

export function UserMessage({ message }: UserMessageProps) {
  return (
    <div className="flex justify-end">
      <div className="max-w-[80%] px-5 py-3 rounded-2xl bg-[#1e5a9a] text-white shadow-sm">
        {message.content}
      </div>
    </div>
  );
}