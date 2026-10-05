import React, { useEffect, useRef } from 'react';

export default function Visualizer({ stream, isActive }: { stream: MediaStream | null, isActive: boolean }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>();
  const analyserRef = useRef<AnalyserNode | null>(null);

  useEffect(() => {
    if (!stream || !canvasRef.current || !isActive) return;

    // Use standard AudioContext
    const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;
    if (!AudioContextClass) return;

    const audioCtx = new AudioContextClass();
    const analyser = audioCtx.createAnalyser();
    analyser.fftSize = 64;
    analyserRef.current = analyser;

    const source = audioCtx.createMediaStreamSource(stream);
    source.connect(analyser);

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d')!;
    
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    const draw = () => {
      animationRef.current = requestAnimationFrame(draw);
      analyser.getByteFrequencyData(dataArray);

      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const barWidth = (canvas.width / bufferLength) * 1.5;
      let x = 0;

      for (let i = 0; i < bufferLength; i++) {
        const barHeight = (dataArray[i] / 255) * canvas.height;
        
        // Draw rounded rect
        ctx.fillStyle = '#06b6d4'; // Cyan-500
        ctx.beginPath();
        ctx.roundRect(x, canvas.height - barHeight, barWidth - 4, Math.max(barHeight, 4), 4);
        ctx.fill();

        x += barWidth;
      }
    };

    draw();

    return () => {
      if (animationRef.current) cancelAnimationFrame(animationRef.current);
      if (audioCtx.state !== 'closed') {
        audioCtx.close().catch(console.error);
      }
    };
  }, [stream, isActive]);

  return (
    <div className="bg-[#f8fafc] rounded-xl sm:rounded-2xl p-4 sm:p-6 shadow-sm border border-slate-100">
      <div className="flex justify-between items-center mb-4 sm:mb-6">
        <h3 className="text-slate-600 font-bold tracking-wider text-xs sm:text-sm">SIGNAL STRENGTH</h3>
        <span className="text-cyan-500 font-mono text-xs sm:text-sm tracking-widest">{isActive ? 'ACTIVE' : 'STANDBY'}</span>
      </div>
      <div className="h-16 sm:h-32 w-full flex items-end justify-center">
        <canvas ref={canvasRef} width={400} height={128} className="w-full h-full object-contain" />
      </div>
    </div>
  );
}
