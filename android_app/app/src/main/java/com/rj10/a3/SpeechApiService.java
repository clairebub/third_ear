package com.rj10.a3;

import android.app.Service;
import android.content.Intent;
import android.os.Binder;
import android.os.IBinder;
import android.support.annotation.Nullable;

/**
 * A bound service to handle Google Speech Api calls.
 *
 * Created on 11/25/17.
 */

public class SpeechApiService extends Service {

    private final IBinder mBinder = new SpeechApiBinder();

    @Nullable
    @Override
    public IBinder onBind(Intent intent) {
        return mBinder;
    }

    public class SpeechApiBinder extends Binder {
        SpeechApiService getService() {
            return SpeechApiService.this;
        }
    }

    public int foo() {
        return 199;
    }
}
