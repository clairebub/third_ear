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

    private static final String TAG = "SpeedService";

    public SpeechService() {
        super("SpeechService");
    }

    @Override
    protected void onHandleIntent(@Nullable Intent intent) {
        Log.w(TAG, "onHandleIntent");

        ResultReceiver rec = intent.getParcelableExtra("receiver");
        String senderName = intent.getStringExtra("sender");

        Bundle b = new Bundle();
        b.putString("something","aziz");
        rec.send(0, b);
    }
}
