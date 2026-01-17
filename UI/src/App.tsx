import { useState, useEffect } from 'react';
import { Sidebar } from './components/Sidebar';
import { WelcomeState } from './components/WelcomeState';
import { ChatInterface } from './components/ChatInterface';
import { ChatInput } from './components/ChatInput';
import { queryRAG, APIError, checkHealth } from './lib/api';
import { WelcomeStateVersionTwo } from './components/WelcomeStateVersionTwo';


export interface Message {
  id: string;
  type: 'user' | 'assistant';
  content: string;
  sources?: Source[];
  timestamp: Date;
}

export interface Source {
  documentName: string;
  pageNumber: number;
  relevance: number;
  excerpt: string;
  fullText?: string;  // Full chunk text
  imagePath?: string;  // Path to extracted image
  isImage?: boolean;  // Whether this is an image source
}

export interface Conversation {
  id: string;
  title: string;
  lastMessage: string;
  timestamp: Date;
}

export default function App() {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(true);
  const [messages, setMessages] = useState<Message[]>([]);
  const [selectedConversationId, setSelectedConversationId] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [backendStatus, setBackendStatus] = useState<'checking' | 'healthy' | 'error'>('checking');
  const [backendError, setBackendError] = useState<string | null>(null);
  const [conversations] = useState<Conversation[]>([]);

  // Check backend health on startup
  useEffect(() => {
    const checkBackendHealth = async () => {
      try {
        const health = await checkHealth();
        if (health.status === 'healthy') {
          setBackendStatus('healthy');
          console.log('Backend connected:', health.vectorDBStats);
        } else {
          setBackendStatus('error');
          setBackendError('Backend is not ready. Please check the server logs.');
        }
      } catch (error) {
        setBackendStatus('error');
        if (error instanceof APIError) {
          setBackendError(error.message);
        } else {
          setBackendError('Cannot connect to backend server on port 8000.');
        }
        console.error('Backend health check failed:', error);
      }
    };

    checkBackendHealth();
  }, []);

  const handleSendMessage = async (content: string) => {
    const userMessage: Message = {
      id: Date.now().toString(),
      type: 'user',
      content,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);

    try {
      // Call the real API
      const response = await queryRAG(content, true);

      // Debug: Log the actual response
      console.log('API Response:', response);
      console.log('Answer:', response.answer);
      console.log('Sources:', response.sources);

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'assistant',
        content: response.answer,
        sources: response.sources,
        timestamp: new Date()
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      // Handle API errors gracefully
      let errorMessage = 'Sorry, I encountered an error while processing your request.';

      if (error instanceof APIError) {
        if (error.statusCode === undefined) {
          errorMessage = 'Cannot connect to the backend server. Please make sure the API is running on port 8000.';
        } else if (error.statusCode === 503) {
          errorMessage = 'The RAG system is not initialized. Please check that the vector database has been created.';
        } else {
          errorMessage = `Error: ${error.details || error.message}`;
        }
      }

      const errorResponseMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'assistant',
        content: errorMessage,
        timestamp: new Date()
      };

      setMessages(prev => [...prev, errorResponseMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSuggestionClick = (suggestion: string) => {
    handleSendMessage(suggestion);
  };

  const handleNewChat = () => {
    setMessages([]);
    setSelectedConversationId(null);
  };

  return (
    <div className="flex h-screen bg-[#fafafa]">
      <Sidebar 
        collapsed={sidebarCollapsed} 
        onToggle={() => setSidebarCollapsed(!sidebarCollapsed)}
        conversations={conversations}
        selectedConversationId={selectedConversationId}
        onNewChat={handleNewChat}
      />
      
      <main className="flex-1 flex flex-col relative">
        {/* Backend status banner */}
        {backendStatus === 'error' && (
          <div className="bg-red-50 border-b border-red-200 px-6 py-3">
            <div className="max-w-3xl mx-auto flex items-center gap-3">
              <div className="flex-shrink-0">
                <svg className="w-5 h-5 text-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                </svg>
              </div>
              <div className="flex-1">
                <p className="text-sm text-red-800 font-medium">Backend Connection Error</p>
                <p className="text-xs text-red-700">{backendError}</p>
              </div>
            </div>
          </div>
        )}

        {messages.length === 0 ? (
          <div className="flex-1 overflow-y-auto">
            <WelcomeStateVersionTwo onSuggestionClick={handleSuggestionClick} onSend={handleSendMessage} isLoading={isLoading} />
          </div>
        ) : (
          <>
            <div className="flex-1 overflow-y-auto">
              <ChatInterface messages={messages} isLoading={isLoading} />
            </div>
            <ChatInput onSend={handleSendMessage} isLoading={isLoading} />
          </>
        )}
      </main>
    </div>
  );
}