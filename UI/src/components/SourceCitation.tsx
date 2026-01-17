import { useState } from 'react';
import { FileText, ChevronDown, ChevronUp } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import type { Source } from '../App';

interface SourceCitationProps {
  source: Source;
}

export function SourceCitation({ source }: SourceCitationProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const relevancePercentage = Math.round(source.relevance * 100);

  // Helper function to convert indented lines to markdown bullets
  const formatAsMarkdown = (text: string): string => {
    const lines = text.split('\n');
    let inList = false;
    let result: string[] = [];

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      const trimmed = line.trim();

      // Check if this is an indented line (potential list item)
      const isIndented = line.match(/^\s{2,}/) && trimmed.length > 0 && !trimmed.startsWith('-') && !trimmed.startsWith('*');

      if (isIndented) {
        // Add blank line before first list item for proper markdown parsing
        if (!inList && result.length > 0 && result[result.length - 1].trim() !== '') {
          result.push('');
        }
        result.push('- ' + trimmed);
        inList = true;
      } else {
        // Add blank line after list for proper markdown parsing
        if (inList && trimmed !== '') {
          result.push('');
        }
        result.push(line);
        inList = false;
      }
    }

    return result.join('\n');
  };

  // Helper function to strip markdown formatting for excerpts
  const stripMarkdown = (text: string): string => {
    return text
      .replace(/\*\*/g, '') // Remove bold markers
      .replace(/\*/g, '')   // Remove italic markers
      .replace(/^#+\s/gm, '') // Remove headers
      .replace(/^\s*[-*+]\s/gm, ''); // Remove bullet points
  };

  return (
    <div
      className="p-4 rounded-xl bg-white border border-black/10 hover:border-[#1e5a9a]/30 transition-colors cursor-pointer"
      onClick={() => setIsExpanded(!isExpanded)}
    >
      <div className="flex items-start gap-3">
        <div className="w-9 h-9 rounded-lg bg-[#1e5a9a]/10 flex items-center justify-center flex-shrink-0">
          <FileText className="w-5 h-5 text-[#1e5a9a]" />
        </div>
        
        <div className="flex-1 min-w-0">
          <div className="flex items-center justify-between gap-3 mb-2">
            <div className="text-sm text-[#1e3a5f] truncate">
              {source.documentName}
            </div>
            <div className="flex items-center gap-2 flex-shrink-0">
              <div className="text-xs text-black/50">
                Page {source.pageNumber}
              </div>
              <div className="flex items-center gap-1">
                <div className="w-12 h-1.5 bg-black/5 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-[#1e5a9a] rounded-full"
                    style={{ width: `${relevancePercentage}%` }}
                  />
                </div>
                <span className="text-xs text-black/40">{relevancePercentage}%</span>
                {isExpanded ? (
                  <ChevronUp className="w-4 h-4 text-black/40 ml-1" />
                ) : (
                  <ChevronDown className="w-4 h-4 text-black/40 ml-1" />
                )}
              </div>
            </div>
          </div>

          <div className="text-sm text-black/60">
            {stripMarkdown(source.excerpt)}
          </div>

          {isExpanded && (
            <div className="mt-3 pt-3 border-t border-black/5 space-y-4">
              {/* Full Text */}
              {source.fullText && (
                <div className="text-xs text-black/70">
                  <div className="font-medium mb-2 mt-1">
                    {source.isImage ? 'Image Description:' : 'Full Text:'}
                  </div>
                  <div className="bg-black/5 px-3 py-2.5 rounded text-xs text-black/75 leading-relaxed max-h-64 overflow-y-auto prose prose-sm max-w-none">
                    {source.isImage ? (
                      <ReactMarkdown
                        components={{
                          h1: ({node, ...props}) => <h1 className="text-sm font-bold mb-2 text-black/80" {...props} />,
                          h2: ({node, ...props}) => <h2 className="text-sm font-bold mb-2 text-black/80" {...props} />,
                          h3: ({node, ...props}) => <h3 className="text-xs font-bold mb-1 text-black/80" {...props} />,
                          strong: ({node, ...props}) => <strong className="font-semibold text-black/80" {...props} />,
                          p: ({node, ...props}) => <p className="mb-2 last:mb-0" {...props} />,
                          ul: ({node, ...props}) => <ul className="list-disc pl-4 mb-2 space-y-1" {...props} />,
                          ol: ({node, ...props}) => <ol className="list-decimal pl-4 mb-2 space-y-1" {...props} />,
                          li: ({node, ...props}) => <li className="text-black/75" {...props} />,
                          code: ({node, ...props}) => <code className="bg-black/10 px-1 rounded text-xs" {...props} />,
                        }}
                      >
                        {formatAsMarkdown(source.fullText)}
                      </ReactMarkdown>
                    ) : (
                      source.fullText
                    )}
                  </div>
                </div>
              )}

              {/* Image Display */}
              {source.isImage && source.imagePath && (
                <div className="text-xs text-black/70">
                  <div className="font-medium mb-2">Extracted Image:</div>
                  <div className="bg-black/5 px-3 py-2.5 rounded">
                    <img
                      src={`http://localhost:8000/images/${source.imagePath.split('/').slice(-1)[0]}`}
                      alt={`Page ${source.pageNumber}`}
                      className="max-w-full h-auto rounded border border-black/10"
                      onError={(e) => {
                        // Fallback if image can't load
                        e.currentTarget.style.display = 'none';
                        const parent = e.currentTarget.parentElement;
                        if (parent) {
                          parent.innerHTML = '<div class="text-black/50 italic">Image not available</div>';
                        }
                      }}
                    />
                  </div>
                </div>
              )}

              {/* Document Location */}
              <div className="text-xs text-black/70">
                <div className="font-medium mb-1">Document Location:</div>
                <code className="block bg-black/5 px-2 py-1 rounded text-xs text-black/80 font-mono">
                  data/sample_docs/{source.documentName.replace('[Image] ', '')}
                </code>
                <div className="text-black/50 italic mt-2">
                  Open this file and navigate to page {source.pageNumber} to see the full context.
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}