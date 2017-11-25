package com.rj10.a3;

import android.os.Bundle;
import android.os.Handler;
import android.os.ResultReceiver;
import android.util.Log;

/**
 * Created by rjtang on 11/25/17.
 */

public class SpeechResultReceiver extends ResultReceiver {
    public SpeechResultReceiver(Handler handler) {
        super(handler);
    }

    @Override
    protected void onReceiveResult(int resultCode, Bundle resultData) {
        Log.d("deebug","received result from Service="+resultData.getString("ServiceTag"));
    }
}
