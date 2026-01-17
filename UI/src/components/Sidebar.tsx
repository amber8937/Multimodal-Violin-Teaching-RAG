import { Plus, Menu } from 'lucide-react';
import type { Conversation } from '../App';

interface SidebarProps {
  collapsed: boolean;
  onToggle: () => void;
  conversations: Conversation[];
  selectedConversationId: string | null;
  onNewChat: () => void;
}

export function Sidebar({ collapsed, onToggle, conversations, selectedConversationId, onNewChat }: SidebarProps) {
  const formatTime = (date: Date) => {
    const now = new Date();
    const diff = now.getTime() - date.getTime();
    const hours = Math.floor(diff / 3600000);
    const days = Math.floor(diff / 86400000);
    
    if (hours < 1) return 'Just now';
    if (hours < 24) return `${hours}h ago`;
    if (days === 1) return 'Yesterday';
    return `${days}d ago`;
  };

  if (collapsed) {
    return (
      <div className="w-16 bg-[#1e3a5f] flex flex-col items-center py-4 px-2">
        {/* Hamburger menu button - pill shape */}
        <div className="relative group">
          <button
            onClick={onToggle}
            className="w-10 h-8 rounded-lg hover:bg-white/10 flex items-center justify-center transition-colors"
            aria-label="Expand sidebar"
          >
            <Menu className="w-[18px] h-[18px] text-white" />
          </button>
          <div className="absolute left-14 top-1/2 -translate-y-1/2 opacity-0 group-hover:opacity-100 pointer-events-none transition-opacity z-50">
            <div className="bg-[#1e3a5f] text-white text-xs px-3 py-1.5 rounded-lg whitespace-nowrap shadow-lg">
              Show sidebar
            </div>
          </div>
        </div>

        {/* Spacing between buttons */}
        <div className="h-3"></div>

        {/* New chat button - rounded rectangle */}
        <div className="relative group">
          <button
            onClick={onNewChat}
            className="w-10 h-9 rounded-lg bg-[#2563eb] hover:bg-[#3b82f6] flex items-center justify-center transition-colors"
            aria-label="New conversation"
          >
            <Plus className="w-5 h-5 text-white" />
          </button>
          <div className="absolute left-14 top-1/2 -translate-y-1/2 opacity-0 group-hover:opacity-100 pointer-events-none transition-opacity z-50">
            <div className="bg-[#1e3a5f] text-white text-xs px-3 py-1.5 rounded-lg whitespace-nowrap shadow-lg">
              New chat
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="w-[260px] bg-[#1e3a5f] border-r border-white/10 flex flex-col">
      {/* Header row */}
      <div className="p-4 flex items-center gap-2">
        <button
          onClick={onToggle}
          className="w-8 h-8 rounded-lg hover:bg-white/10 flex items-center justify-center transition-colors flex-shrink-0"
          aria-label="Collapse sidebar"
        >
          <Menu className="w-[18px] h-[18px] text-white" />
        </button>
        <button
          onClick={onNewChat}
          className="flex-1 h-10 px-3 rounded-lg bg-[#2563eb] hover:bg-[#3b82f6] text-white flex items-center justify-center gap-2 transition-colors font-medium"
        >
          <Plus className="w-5 h-5" />
          <span className="text-sm">New Chat</span>
        </button>
      </div>

      {/* Conversations list */}
      <div className="flex-1 overflow-y-auto px-2 pb-4">
        <div className="text-[11px] text-white/50 px-4 pt-5 pb-3 uppercase tracking-wide">
          Recent Conversations
        </div>
        {conversations.map((conv) => (
          <button
            key={conv.id}
            className="w-full px-4 py-3 mx-2 rounded-lg hover:bg-white/8 text-left transition-colors group mb-1"
            style={{ width: 'calc(100% - 16px)' }}
          >
            <div className="flex-1 min-w-0">
              <div className="text-sm text-white mb-1 truncate font-medium">
                {conv.title}
              </div>
              <div className="text-xs text-white/50 truncate">
                {conv.lastMessage}
              </div>
              <div className="text-[11px] text-white/40 mt-1">
                {formatTime(conv.timestamp)}
              </div>
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}