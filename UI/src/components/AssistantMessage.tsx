import { useState } from 'react';
import { Copy, ThumbsUp, ThumbsDown } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { SourceCitation } from './SourceCitation';
import type { Message } from '../App';

interface AssistantMessageProps {
  message: Message;
}

export function AssistantMessage({ message }: AssistantMessageProps) {
  const [copied, setCopied] = useState(false);
  const [thumbsUpActive, setThumbsUpActive] = useState(false);
  const [thumbsDownActive, setThumbsDownActive] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleThumbsUp = () => {
    if (thumbsUpActive) {
      setThumbsUpActive(false);
    } else {
      setThumbsUpActive(true);
      setThumbsDownActive(false);
    }
  };

  const handleThumbsDown = () => {
    if (thumbsDownActive) {
      setThumbsDownActive(false);
    } else {
      setThumbsDownActive(true);
      setThumbsUpActive(false);
    }
  };

  return (
    <div className="flex justify-start">
      <div className="max-w-[85%] space-y-4">
        <div className="py-4">
          <div className="prose prose-slate max-w-none">
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              components={{
                h1: ({ children }) => (
                  <h1 className="text-2xl text-[#1e5a9a] mb-6 mt-14 first:mt-0 font-bold leading-tight">
                    {children}
                  </h1>
                ),
                h2: ({ children }) => (
                  <h2 className="text-xl text-[#1e5a9a] mb-5 mt-12 first:mt-0 font-bold leading-snug">
                    {children}
                  </h2>
                ),
                h3: ({ children }) => (
                  <h3 className="text-lg text-[#1e5a9a] mb-4 mt-10 first:mt-0 font-semibold leading-snug">
                    {children}
                  </h3>
                ),
                h4: ({ children }) => (
                  <h4 className="text-[16px] text-[#1e5a9a] mb-4 mt-9 font-semibold leading-normal">
                    {children}
                  </h4>
                ),
                p: ({ children }) => (
                  <p className="text-[15px] text-black/80 mb-6 leading-[1.8]">
                    {children}
                  </p>
                ),
                ul: ({ children }) => (
                  <ul className="list-disc list-outside ml-8 mb-6 space-y-3 [&_ul]:ml-6 [&_ul]:mt-2 [&_ul]:mb-2">
                    {children}
                  </ul>
                ),
                ol: ({ children }) => (
                  <ol className="list-decimal list-outside ml-8 mb-6 space-y-3 [&_ol]:ml-6 [&_ol]:mt-2 [&_ol]:mb-2 [&_ul]:ml-6 [&_ul]:mt-2 [&_ul]:mb-2">
                    {children}
                  </ol>
                ),
                li: ({ children }) => (
                  <li className="text-[15px] text-black/85 leading-[1.8] pl-2 mb-2">
                    {children}
                  </li>
                ),
                strong: ({ children }) => (
                  <strong className="font-bold text-black/95">{children}</strong>
                ),
                em: ({ children }) => <em className="italic text-black/75">{children}</em>,
                code: ({ children }) => (
                  <code className="bg-black/5 px-2 py-0.5 rounded text-sm font-mono text-black/85">
                    {children}
                  </code>
                ),
                hr: () => <hr className="border-black/10" style={{ marginTop: '2rem', marginBottom: '2rem' }} />,
                blockquote: ({ children }) => (
                  <blockquote className="border-l-4 border-[#1e5a9a]/30 pl-6 italic text-black/70 my-6 py-2">
                    {children}
                  </blockquote>
                ),
              }}
            >
              {message.content}
            </ReactMarkdown>
          </div>
          
          <div className="flex items-center gap-2 mt-6 pt-4 border-t border-black/10">
            <button
              onClick={handleCopy}
              className="p-2 rounded-full hover:bg-black/5 transition-colors group"
              aria-label="Copy message"
            >
              <Copy className={`w-4 h-4 ${copied ? 'text-[#1e5a9a]' : 'text-[#666] group-hover:text-[#1e3a5f]'}`} />
            </button>
            <button
              onClick={handleThumbsUp}
              className="p-2 rounded-full hover:bg-black/5 transition-colors"
              aria-label="Good response"
            >
              <ThumbsUp className={`w-4 h-4 ${thumbsUpActive ? 'text-[#1e5a9a] fill-[#1e5a9a]' : 'text-[#666] hover:text-[#1e3a5f]'}`} />
            </button>
            <button
              onClick={handleThumbsDown}
              className="p-2 rounded-full hover:bg-black/5 transition-colors"
              aria-label="Bad response"
            >
              <ThumbsDown className={`w-4 h-4 ${thumbsDownActive ? 'text-[#1e5a9a] fill-[#1e5a9a]' : 'text-[#666] hover:text-[#1e3a5f]'}`} />
            </button>
            {copied && (
              <span className="text-xs text-[#1e5a9a] ml-1">Copied!</span>
            )}
          </div>
        </div>

        {message.sources && message.sources.length > 0 && (
          <div className="space-y-2">
            <div className="text-xs text-black/50 px-1">Sources</div>
            {message.sources.map((source, index) => (
              <SourceCitation key={index} source={source} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}