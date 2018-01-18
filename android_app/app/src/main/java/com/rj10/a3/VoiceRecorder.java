package com.rj10.a3;

import android.support.annotation.NonNull;

import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.telecom.Call;
import android.util.Log;

/**
 * An object that utilized the AudioRecord interface to access android's audio records. It used
 * a separate worker thread to continuously poll the AudioRecord's buffer and do something with
 * the audio data.
 */

public class VoiceRecorder {

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


    private static final String TAG = "VoiceRecorder";
    // private static final int[] SAMPLE_RATE_CANDIDATES = new int[]{16000, 11025, 22050, 44100};
    private static final int[] SAMPLE_RATE_CANDIDATES = new int[]{ 22050, 44100};
    private static final int CHANNEL = AudioFormat.CHANNEL_IN_MONO;
    private static final int ENCODING = AudioFormat.ENCODING_PCM_16BIT;
    private static final int AMPLITUDE_THRESHOLD = 1500;
    private static final int SPEECH_TIMEOUT_MILLIS = 2000;
    private static final int MAX_SPEECH_LENGTH_MILLIS = 30 * 1000;

    private Thread mThread;

    public VoiceRecorder() {}

    /**
     * Starts recording audio.
     *
     * <p>The caller is responsible for calling {@link #stop()} later.</p>
     */
    public void start(@NonNull Callback callback) {
        mThread = new Thread(new VoiceRecorderRunnable(callback));
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
    private class VoiceRecorderRunnable implements Runnable {
        private final Callback mCallback;
        private AudioRecord mAudioRecord;
        private byte[] mBuffer;
        /** The timestamp of the last time that voice is heard. */
        private long mLastVoiceHeardMillis = Long.MAX_VALUE;
        /** The timestamp when the current voice is started. */
        private long mVoiceStartedMillis;

        public VoiceRecorderRunnable(Callback callback) {
            mCallback = callback;

            // Try to create a new recording session.
            mAudioRecord = createAudioRecord();
            if (mAudioRecord == null) {
                // TODO: show a status on the associated UI so the user knows
                throw new RuntimeException("Cannot instantiate VoiceRecorder");
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
            for (int sampleRate : SAMPLE_RATE_CANDIDATES) {
                final int minSizeInBytes = AudioRecord.getMinBufferSize(sampleRate, CHANNEL, ENCODING);
                if (minSizeInBytes == AudioRecord.ERROR_BAD_VALUE) {
                    continue;
                }
                int sizeInBytes = 1 * sampleRate * 2; // buffer should at least hold one seconds of samples
                if (sizeInBytes < minSizeInBytes) {
                    sizeInBytes = minSizeInBytes;
                }
                final AudioRecord audioRecord = new AudioRecord(MediaRecorder.AudioSource.MIC,
                        sampleRate, CHANNEL, ENCODING, sizeInBytes);
                if (audioRecord.getState() == AudioRecord.STATE_INITIALIZED) {
                    mBuffer = new byte[sizeInBytes];
                    return audioRecord;
                } else {
                    audioRecord.release();
                }
            }
            return null;
        }

        @Override
        public void run() {
            while (true) {
                if (Thread.currentThread().isInterrupted()) {
                    Log.d(TAG, "ProcessVoiceRunnable thread interrupted: " + Thread.currentThread());
                    dismiss();
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
                        end();
                    }
                } else if (mLastVoiceHeardMillis != Long.MAX_VALUE) {
                    mCallback.onVoice(mBuffer, size);
                    if (now - mLastVoiceHeardMillis > SPEECH_TIMEOUT_MILLIS) {
                        end();
                    }
                }
                //}
            }
        }

        private void end() {
            mLastVoiceHeardMillis = Long.MAX_VALUE;
            mCallback.onVoiceEnd();
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
