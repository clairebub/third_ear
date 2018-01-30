package com.rj10.a3;

import android.os.Environment;
import android.support.annotation.NonNull;

import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.telecom.Call;
import android.util.Log;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOError;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.text.SimpleDateFormat;
import java.util.Date;

/**
 * An object that utilized the AudioRecord interface to access android's audio records. It used
 * a separate worker thread to continuously poll the AudioRecord's buffer and do something with
 * the audio data.
 */

public class SoundRecorder {

    interface Callback {
        /**
         * Called when the recorder starts hearing voice.
         */
        void onVoiceStart(int sampleRate);

        /**
         * Called when the recorder is hearing voice.
         *
         * @param data The audio data in {@link AudioFormat#ENCODING_PCM_16BIT}.
         * @param size The size of the actual data in {@code data}.
         */
        void onVoice(byte[] data, int size);

        /**
         * Called when the recorder stops hearing voice.
         */
        void onVoiceEnd();

        /**
         * mainly used for debugging to pass the status message.
         * @param msg
         */
        void deebug(String msg);
    }


    private static final String TAG = "SoundRecorder";
    private static final int SAMPLE_RATE = 22050;
    private static final int AUDIO_CHANNEL = AudioFormat.CHANNEL_IN_MONO;
    private static final int AUDIO_ENCODING = AudioFormat.ENCODING_PCM_16BIT;
    private static final int AMPLITUDE_THRESHOLD = 1500;
    private static final int SPEECH_TIMEOUT_MILLIS = 2000;
    private static final int MAX_SPEECH_LENGTH_MILLIS = 30 * 1000;

    private Thread mThread;

    public SoundRecorder() {}

    /**
     * Starts recording audio.
     *
     * <p>The caller is responsible for calling {@link #stop()} later.</p>
     */
    public void start(@NonNull Callback callback) {
        if (mThread != null) {
            throw new IllegalStateException();
        }
        mThread = new Thread(new SoundRecorderRunnable(callback));
        mThread.start();
    }

    /**
     * Stops recording audio.
     */
    public void stop() {
        if (mThread != null) {
            mThread.interrupt();
            mThread = null;
        }
    }

    /**
     * A worker thread that extracts buffer from AudioRecord and invokes the supplied callback for
     * speech recognition.
     */
    private class SoundRecorderRunnable implements Runnable {
        private final Callback mCallback;
        private AudioRecord mAudioRecord;
        private byte[] mBuffer;
        private ByteBuffer mAudioBytes;
        /** The timestamp of the last time that voice is heard. */
        private long mLastVoiceHeardMillis = Long.MAX_VALUE;
        /** The timestamp when the current voice is started. */
        private long mVoiceStartedMillis;

        public SoundRecorderRunnable(Callback callback) {
            mCallback = callback;

            // Try to create a new recording session.
            mAudioRecord = createAudioRecord();
            if (mAudioRecord == null) {
                // TODO: show a status on the associated UI so the user knows
                throw new RuntimeException("Cannot instantiate SoundRecorder");
            }
            // Start recording.
            mAudioRecord.startRecording();
        }

        /**
         * Dismisses the currently ongoing utterance and releases the resources
         */
        private void dismiss() {
            if (mLastVoiceHeardMillis != Long.MAX_VALUE) {
                mLastVoiceHeardMillis = Long.MAX_VALUE;
                mCallback.onVoiceEnd();
            }
            if (mAudioRecord != null) {
                mAudioRecord.stop();
                mAudioRecord.release();
                mAudioRecord = null;
            }
            mBuffer = null;
        }

        /**
         * Creates a new {@link AudioRecord}.
         *
         * @return A newly created {@link AudioRecord}, or null if it cannot be created (missing
         * permissions?).
         */
        private AudioRecord createAudioRecord() {
            int minBufferSizeInBytes = getMinBufferSizeInBytes();
            AudioRecord audioRecord = new AudioRecord(
                    MediaRecorder.AudioSource.MIC,
                    SAMPLE_RATE,
                    AUDIO_CHANNEL, AUDIO_ENCODING,
                    minBufferSizeInBytes);
            if (audioRecord.getState() != AudioRecord.STATE_INITIALIZED) {
                audioRecord.release();
                throw new IllegalStateException();
            }
            mBuffer = new byte[minBufferSizeInBytes];
            // initial capacity to hold 10 seconds of audio
            mAudioBytes = ByteBuffer.allocate(MAX_SPEECH_LENGTH_MILLIS/1000*SAMPLE_RATE*2);
            return audioRecord;
        }

        private int getMinBufferSizeInBytes() {
            int minBufferSize = AudioRecord.getMinBufferSize(
                SAMPLE_RATE,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT);
            Log.d(TAG, "AudioRecord.getMinBufferSize() returns " + minBufferSize);
            // make sure minBufferSize can contain at least 1 second of audio (16 bits sample).
            if (minBufferSize < SAMPLE_RATE * 2) {
                minBufferSize = SAMPLE_RATE * 2;
                Log.d(TAG, "minBufferSize increased to store 1 sec of audio: " + minBufferSize);
            }
            return minBufferSize;
        }

        @Override
        public void run() {
            while (true) {
                if (Thread.currentThread().isInterrupted()) {
                    Log.d(TAG, "ProcessVoiceRunnable thread interrupted: " + Thread.currentThread());
                    break;
                }
                final int size = mAudioRecord.read(mBuffer, 0, mBuffer.length);
                // TODO: need to check if size is negative
                if (size <= 0) {
                    continue;
                }
                final long now = System.currentTimeMillis();
                if (isHearingVoice(mBuffer, size)) {
                    if (mLastVoiceHeardMillis == Long.MAX_VALUE) {
                        mVoiceStartedMillis = now;
                        mCallback.onVoiceStart(mAudioRecord.getSampleRate());
                    }
                    mCallback.onVoice(mBuffer, size);
                    mLastVoiceHeardMillis = now;
                    if (now - mVoiceStartedMillis > MAX_SPEECH_LENGTH_MILLIS) {
                        Log.d(TAG, "recording sound over max length: " + MAX_SPEECH_LENGTH_MILLIS);
                        end();
                        break;
                    }
                    if (mAudioBytes.remaining() <= 0) {
                        Log.d(TAG, "AudioBytes full");
                        end();
                        break;
                    }
                    mAudioBytes.put(mBuffer, 0, size);
                } else if (mLastVoiceHeardMillis != Long.MAX_VALUE) {
                    mCallback.onVoice(mBuffer, size);
                    if (now - mLastVoiceHeardMillis > SPEECH_TIMEOUT_MILLIS) {
                        Log.d(TAG, "recording timeout " + SPEECH_TIMEOUT_MILLIS);
                        end();
                        break;
                    }
                }
            }
            // write the WAV file
            try {
                writeWAVFile();
            } catch (IOException ex) {
                Log.e(TAG, ex.toString());
            }
            dismiss();
        }

        private void end() {
            mLastVoiceHeardMillis = Long.MAX_VALUE;
            mCallback.onVoiceEnd();
        }

        private void writeWAVFile() throws IOException {
            int numSamples = mAudioBytes.position() / 2;
            if (numSamples <= SAMPLE_RATE) {
                Log.d(TAG, "not enough samples for 1 sec: " + numSamples);
                return;
            }
            String externalRootDir = Environment.getExternalStorageDirectory().getPath();
            if (!externalRootDir.endsWith("/")) {
                externalRootDir += "/";
            }
            File parentDir = new File(externalRootDir + "third_ear/");
            parentDir.mkdirs();
            String fileName =
                    new SimpleDateFormat("yyyyMMddHHmmss'.wav'").format(new Date());
            File wavFile = new File(parentDir, fileName);
            Log.d(TAG, "wav file " + wavFile);
            FileOutputStream outputStream = new FileOutputStream(wavFile);

            outputStream.write(WAVHeader.getWAVHeader(SAMPLE_RATE, 1, numSamples));
            Log.d(TAG, "writing WAV file with " + numSamples + " samples");

            mAudioBytes.position(0);
            int nBytesLeft = numSamples * 2;
            while (nBytesLeft > 0) {
                // write the samples to the file, mBuffer length bytes at a time
                int nBytesRead = mBuffer.length;
                if (nBytesLeft < mBuffer.length) {
                    nBytesRead = nBytesLeft;
                }
                mAudioBytes.get(mBuffer, 0, nBytesRead);
                outputStream.write(mBuffer, 0, nBytesRead);
                nBytesLeft -= nBytesRead;
            }
            outputStream.close();
        }

        private boolean isHearingVoice(byte[] buffer, int size) {
            for (int i = 0; i < size - 1; i += 2) {
                // The buffer has LINEAR16 in little endian.
                int s = buffer[i + 1];
                if (s < 0) s *= -1;
                s <<= 8;
                s += Math.abs(buffer[i]);
                if (s > AMPLITUDE_THRESHOLD) {
                    return true;
                }
            }
            return false;
        }
    }
}
