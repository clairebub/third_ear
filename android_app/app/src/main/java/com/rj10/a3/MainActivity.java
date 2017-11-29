package com.rj10.a3;

import android.Manifest;
import android.content.ActivityNotFoundException;
import android.content.ComponentName;
import android.content.Context;
import android.content.Intent;
import android.content.ServiceConnection;
import android.content.pm.PackageManager;
import android.os.Handler;
import android.os.IBinder;
import android.os.ResultReceiver;
import android.speech.RecognizerIntent;
import android.support.annotation.NonNull;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.RadioButton;
import android.widget.RadioGroup;
import android.widget.TextView;
import android.widget.Toast;

import java.util.ArrayList;
import java.util.Locale;

public class MainActivity extends AppCompatActivity
        implements ActivityCompat.OnRequestPermissionsResultCallback {

    private final int REQ_CODE_SPEECH_RECOG_LOCAL = 100;
    private final int PERMISSION_REQUEST_READ_PHONE_STATE = 1;
    private final int PERMISSION_REQUEST_RECORD_AUDIO = 2;

    Button mOnButton;
    TextView mStatus;
    RadioGroup mRadioGroup;

    SpeechResultReceiver speechResultReceiver = null;
    private VoiceRecorder mVoiceRecorder;
    private SpeechService mSpeechService;
    private SpeechApiService mSpeechApiService;

    int mRequestCount = 0;

    //
    // Callback from voice recorder to handle start/ongoing/end of voice input.
    // For cloud based speech recog streaming Api only.
    //
    private final VoiceRecorder.Callback mVoiceCallback = new VoiceRecorder.Callback() {
        @Override
        public void onVoiceStart() {
            showStatus("onVoiceStart");
            if (mSpeechService != null) {
                mSpeechService.startRecognizing(mVoiceRecorder.getSampleRate());
            }
        }

        @Override
        public void onVoice(byte[] data, int size) {
            showStatus("onVoice");
            if (mSpeechService != null) {
                mSpeechService.recognize(data, size);
            }
        }

        @Override
        public void onVoiceEnd() {
            showStatus("onVoiceEnd");
            if (mSpeechService != null) {
                mSpeechService.finishRecognizing();
            }
        }

        @Override
        public void onStatusUpdate(String msg) {
            showStatus(msg);
        }
    };


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        speechResultReceiver = new SpeechResultReceiver(new Handler());

        mStatus = (TextView) findViewById(R.id.status);
        mStatus.setHint("Version: 0.1c");
        mOnButton = (Button) findViewById(R.id.buttonOn);
        mOnButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                handleOnClick();
            }
        });

        mRadioGroup = (RadioGroup) findViewById(R.id.radioGroup);
        mRadioGroup.check(R.id.radioButton_local);
        mRadioGroup.setOnCheckedChangeListener(new RadioGroup.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(RadioGroup radioGroup, int checkedId) {
                RadioButton rb = (RadioButton) radioGroup.findViewById(checkedId);
                if (null != rb && checkedId > -1) {
                    Toast.makeText(MainActivity.this, rb.getText(), Toast.LENGTH_SHORT).show();
                }
            }
        });
    }

    @Override
    protected void onStart() {
        super.onStart();
        // Bind to SpeechApiService
        Intent intent = new Intent(this, SpeechApiService.class);
        bindService(intent, mSpeechApiConnection, Context.BIND_AUTO_CREATE);
    }

    private void startSpeechRecLocal() {
        Intent intent = new Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH);
        intent.putExtra(
                RecognizerIntent.EXTRA_LANGUAGE_MODEL,
                RecognizerIntent.LANGUAGE_MODEL_FREE_FORM);
        intent.putExtra(RecognizerIntent.EXTRA_LANGUAGE, Locale.getDefault());
        intent.putExtra(
                RecognizerIntent.EXTRA_PROMPT,
                "Starting speech recognition...");
        try {
            startActivityForResult(intent, REQ_CODE_SPEECH_RECOG_LOCAL);
        } catch (ActivityNotFoundException a) {
            deebug(a.toString());
        }
    }

    private void startSpeechRecCloud() {
        checkPermissions(Manifest.permission.RECORD_AUDIO, PERMISSION_REQUEST_RECORD_AUDIO);

        if (mVoiceRecorder != null) {
            mVoiceRecorder.stop();
        }
        mVoiceRecorder = new VoiceRecorder(mVoiceCallback);
        mVoiceRecorder.start();

        Intent intent = new Intent(this, SpeechService.class);
        intent.putExtra("receiver", speechResultReceiver);
        intent.putExtra("sender", "stem main");
        startService(intent);
    }

    private void handleOnClick() {
        mRequestCount++;
        String msg = String.format("Starting request #%d", mRequestCount);
        msg = "foo: " + mSpeechApiService.foo();
        mStatus.setText(msg);
        // startSpeechRec();

        int radioButtonID = mRadioGroup.getCheckedRadioButtonId();
        if (radioButtonID == R.id.radioButton_local) {
            startSpeechRecLocal();
        } else {
            deebug("unexpected radioButtonId: " + radioButtonID);
        }

    }
    private void stopSpeechRec() {
        if (mVoiceRecorder != null) {
            mVoiceRecorder.stop();
            mVoiceRecorder = null;
        }
    }

    /*
    void startSpeedRec() {
        checkPermissions(Manifest.permission.READ_PHONE_STATE, PERMISSION_REQUEST_READ_PHONE_STATE);
        checkPermissions(Manifest.permission.RECORD_AUDIO, PERMISSION_REQUEST_RECORD_AUDIO);

        Intent intent = new Intent(this, SpeechService.class);
        intent.putExtra("receiver", speechResultReceiver);
        intent.putExtra("sender", "stem main");
        startService(intent);

    }
*/
    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        switch (requestCode) {
            case REQ_CODE_SPEECH_RECOG_LOCAL: {
                if (resultCode == RESULT_OK && null != data) {
                    ArrayList<String> result = data
                            .getStringArrayListExtra(RecognizerIntent.EXTRA_RESULTS);
                   deebug(result.get(0));
                }
                break;
            }
            default:
                deebug("unexpected request code: " + requestCode);
                break;
        }
    }

    public class SpeechResultReceiver extends ResultReceiver {
        public SpeechResultReceiver(Handler handler) {
            super(handler);
        }

        @Override
        protected void onReceiveResult(int resultCode, Bundle resultData) {
            final String result = resultData.getString(SpeechService.MSG);
            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    String msg = String.format("%d: %s", mRequestCount, result);
                    mStatus.setText(msg);
                }
            });
        }
    }

    private void checkPermissions(String permission, int requestCode) {
        if (ContextCompat.checkSelfPermission(this, permission)
                != PackageManager.PERMISSION_GRANTED) {
            Log.d("deebug", "requesting permission for " + permission);
            ActivityCompat.requestPermissions(this,
                    new String[]{permission},
                    requestCode);
        }
        Log.d("deebug", "has permission: " + permission);
    }

    @Override
    public void onRequestPermissionsResult(
            int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        if (requestCode == PERMISSION_REQUEST_READ_PHONE_STATE) {
            if (grantResults.length == 1 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                // all good
            } else {
                Toast.makeText(this,
                        "The app won't read phone state.", Toast.LENGTH_LONG).show();
            }
        }  else if (requestCode == PERMISSION_REQUEST_RECORD_AUDIO) {
            if (grantResults.length == 1 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                // all good
            } else {
                Toast.makeText(this,
                        "The app won't record audio.", Toast.LENGTH_LONG).show();
            }
        } else {
            super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        }
    }



    /** Defines callbacks for service binding, passed to bindService() */
    private final ServiceConnection mSpeechApiConnection = new ServiceConnection() {
        @Override
        public void onServiceConnected(ComponentName className,
                                       IBinder service) {
            // We've bound to LocalService, cast the IBinder and get LocalService instance
            SpeechApiService.SpeechApiBinder binder = (SpeechApiService.SpeechApiBinder) service;
            mSpeechApiService = binder.getService();
        }

        @Override
        public void onServiceDisconnected(ComponentName arg0) {
            mSpeechApiService = null;
        }
    };

    void showStatus(final String msg) {
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                mStatus.setText(msg);
            }
        });
    }

    private void deebug(final String msg) {
        Log.d("deebug", msg);
        Toast.makeText(getApplicationContext(), msg, Toast.LENGTH_SHORT).show();
    }
}
