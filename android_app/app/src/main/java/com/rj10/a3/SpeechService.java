package com.rj10.a3;

import android.app.IntentService;
import android.content.Intent;
import android.os.Bundle;
import android.os.ResultReceiver;
import android.support.annotation.Nullable;
import android.util.Log;

/**
 * Created by rjtang on 11/25/17.
 */

public class SpeechService extends IntentService {
    public static final String MSG = "msg";
    private static final String TAG = "SpeedService";

    public SpeechService() {
        super("SpeechService");
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
