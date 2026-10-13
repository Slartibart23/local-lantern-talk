// Captures microphone audio in the audio rendering thread.
// Posts 40 ms blocks of 16-bit PCM at the context's native sample rate (full quality,
// the backend downsamples a copy for VAD / Whisper) plus an RMS level for the meter.

class MicCapture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.blockSize = Math.round(sampleRate * 0.04);
    this.buffer = new Int16Array(this.blockSize);
    this.filled = 0;
    this.sumSquares = 0;
  }

  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (!channel) return true;
    for (let i = 0; i < channel.length; i++) {
      const s = Math.max(-1, Math.min(1, channel[i]));
      this.buffer[this.filled++] = s < 0 ? s * 32768 : s * 32767;
      this.sumSquares += s * s;
      if (this.filled === this.blockSize) {
        const rms = Math.sqrt(this.sumSquares / this.blockSize);
        this.port.postMessage({ pcm: this.buffer.buffer, rms }, [this.buffer.buffer]);
        this.buffer = new Int16Array(this.blockSize);
        this.filled = 0;
        this.sumSquares = 0;
      }
    }
    return true;
  }
}

registerProcessor("mic-capture", MicCapture);
