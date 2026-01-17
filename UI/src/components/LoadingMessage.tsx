import { useEffect, useState } from 'react';

const STAGES = [
  'Embedding query',
  'Searching documents',
  'Reranking results',
  'Generating answer'
];

export function LoadingMessage() {
  const [dots, setDots] = useState('');
  const [stageIndex, setStageIndex] = useState(0);

  useEffect(() => {
    // Animate dots
    const dotsInterval = setInterval(() => {
      setDots((prev) => {
        if (prev === '...') return '';
        return prev + '.';
      });
    }, 400);

    // Cycle through stages (stop at the last stage)
    const stageInterval = setInterval(() => {
      setStageIndex((prev) => {
        if (prev < STAGES.length - 1) {
          return prev + 1;
        }
        return prev; // Stay on "Generating answer"
      });
    }, 6000);

    return () => {
      clearInterval(dotsInterval);
      clearInterval(stageInterval);
    };
  }, []);

  return (
    <div className="flex justify-start">
      <div className="max-w-[85%]">
        <div className="py-4">
          <div className="text-[15px] text-black/60 leading-[1.8]">
            {STAGES[stageIndex]} {dots}
          </div>
        </div>
      </div>
    </div>
  );
}
