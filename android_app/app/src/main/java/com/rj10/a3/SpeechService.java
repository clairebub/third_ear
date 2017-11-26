package com.rj10.a3;

import android.app.IntentService;
import android.content.Intent;
import android.os.Bundle;
import android.os.ResultReceiver;
import android.support.annotation.Nullable;
import android.util.Log;

/**
 * Created on 11/25/17.
 */

public class SpeechService extends IntentService {
    public static final String MSG = "msg";
    private static final String TAG = "SpeedService";

    public interface Listener {

        /**
         * Called when a new piece of text was recognized by the Speech API.
         *
         * @param text    The text.
         * @param isFinal {@code true} when the API finished processing audio.
         */
        void onSpeechRecognized(String text, boolean isFinal);

    }

    public SpeechService() {
        super("SpeechService");
    }

    /**
     * Starts recognizing speech audio.
     *
     * @param sampleRate The sample rate of the audio.
     */
    public void startRecognizing(int sampleRate) {
        Log.d("deebug", "startRecog with sampleRate=" + sampleRate);
        // TODO
    }

    /**
     * Recognizes the speech audio. This method should be called every time a chunk of byte buffer
     * is ready.
     *
     * @param data The audio data.
     * @param size The number of elements that are actually relevant in the {@code data}.
     */
    public void recognize(byte[] data, int size) {
        Log.d("deebug", "recog with bufsize: " + size);
    }
    /**
     * Finishes recognizing speech audio.
     */
    public void finishRecognizing() {
        Log.d("deebug", "finishRecog");
    }

    @Override
    protected void onHandleIntent(@Nullable Intent intent) {
        Log.w(TAG, "onHandleIntent");

        ResultReceiver rec = intent.getParcelableExtra("receiver");
        String senderName = intent.getStringExtra("sender");

        try {
            Thread.currentThread().sleep(3000);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        Bundle b = new Bundle();
        b.putString(MSG,"aziz 2");
        rec.send(0, b);
    }
}
