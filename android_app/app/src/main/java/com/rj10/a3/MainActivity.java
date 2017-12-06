package com.rj10.a3;

import android.Manifest;
import android.content.ActivityNotFoundException;
import android.content.ComponentName;
import android.content.Intent;
import android.content.ServiceConnection;
import android.content.pm.PackageManager;
import android.os.IBinder;
import android.speech.RecognizerIntent;
import android.support.annotation.NonNull;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.RadioGroup;
import android.widget.TextView;
import android.widget.Toast;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

//
// An app that monitors the ambient sound and do speech detection and outputs the results
// on the screen.
// The UI is the following:
// 1. a button to start/stop the speech detection and indicate whether it's running
// 2. an output area to display the detected speech
//
public class MainActivity extends AppCompatActivity
        implements ActivityCompat.OnRequestPermissionsResultCallback {

    private final int PERMISSIONS_REQUEST_FOR_CLOUD = 1;

    private final int REQ_CODE_SPEECH_RECOG_LOCAL = 100;

    TextView mStatus;
    RadioGroup mRadioGroup;

    private VoiceRecorder mVoiceRecorder;
    private SpeechApiService mSpeechApiService;
    int mRequestCount = 0;

    /** Defines callbacks for service binding, passed to bindService() */
    private final ServiceConnection mSpeechApiServiceConnection = new ServiceConnection() {
        @Override
        public void onServiceConnected(ComponentName className,
                                       IBinder service) {
            // We've bound to LocalService, cast the IBinder and get LocalService instance
            SpeechApiService.SpeechApiBinder binder = (SpeechApiService.SpeechApiBinder) service;
            mSpeechApiService = binder.getService();
            deebug("SpeechApiService connected: " + mSpeechApiService);
        }

        @Override
        public void onServiceDisconnected(ComponentName arg0) {
            mSpeechApiService = null;
            deebug("SpeechApiService disconnected." );
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        mStatus = (TextView) findViewById(R.id.status);
        mStatus.setHint("Version: 0.1c");
        Button onOffButton = (Button) findViewById(R.id.buttonOn);
        onOffButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                mRequestCount++;
                handleOnClick((Button) v);
            }
        });

        mRadioGroup = (RadioGroup) findViewById(R.id.radioGroup);
        mRadioGroup.check(R.id.radioButton_cloud);
        mRadioGroup.setOnCheckedChangeListener(new RadioGroup.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(RadioGroup radioGroup, int checkedId) {
                if (checkedId == R.id.radioButton_local) {
                    // cloud recognition runs in the background. need to make sure it is no
                    // longer running if mode is switching to local
                    stopSpeechRecCloud();
                }
                /*
                RadioButton rb = (RadioButton) radioGroup.findViewById(checkedId);
                if (null != rb && checkedId > -1) {
                    // Toast.makeText(MainActivity.this, rb.getText(), Toast.LENGTH_SHORT).show();
                } */
            }
        });
    }

    /**
     * Bind to the speech api service only when the activity is in use. You can do it at OnStart()
     * or at onResume()
     */
    @Override
    protected void onStart() {
        super.onStart();
        // Prepare Cloud Speech API
        bindService(new Intent(this, SpeechApiService.class),
                mSpeechApiServiceConnection, BIND_AUTO_CREATE);
    }

    private void handleOnClick(Button button) {
        CharSequence buttonText = button.getText();
        deebug("button text: " + buttonText);
        if (buttonText.toString().equalsIgnoreCase("start")) {
            button.setText("Stop");
            Toast.makeText(getApplicationContext(), "Stop", Toast.LENGTH_SHORT).show();
            stopSpeechRec();
        } else {
            button.setText("Start");
            Toast.makeText(getApplicationContext(), "Start", Toast.LENGTH_SHORT).show();
            startSpeechRec();
        }
    }

    private void startSpeechRec() {
        int radioButtonID = mRadioGroup.getCheckedRadioButtonId();
        switch (radioButtonID) {
            case R.id.radioButton_local:
                startSpeechRecLocal();
                break;
            case R.id.radioButton_cloud:
                startSpeechRecCloud();
                break;
            default:
                deebug("unexpected radioButtonId: " + radioButtonID);
                break;
        }
    }

    private void stopSpeechRec() {
        int radioButtonID = mRadioGroup.getCheckedRadioButtonId();
        switch (radioButtonID) {
            case R.id.radioButton_cloud:
                stopSpeechRecCloud();
                break;
            default:
                deebug("unexpected radioButtonId: " + radioButtonID);
                break;
        }
    }

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
        String[] permissions = new String[] {
                Manifest.permission.INTERNET,
                Manifest.permission.RECORD_AUDIO,
                Manifest.permission.READ_PHONE_STATE
        };
        if (!checkPermissions(permissions, PERMISSIONS_REQUEST_FOR_CLOUD)) {
            return;
        }

        if (mVoiceRecorder != null) {
            mVoiceRecorder.stop();
        }
        mVoiceRecorder = new VoiceRecorder(mVoiceCallback);
        mVoiceRecorder.start();
    }


    private void stopSpeechRecCloud() {
        if (mVoiceRecorder != null) {
            mVoiceRecorder.stop();
            mVoiceRecorder = null;
        }
    }

    //
    // Callback from voice recorder to handle start/ongoing/end of voice input.
    // For cloud based speech recog streaming Api only.
    // The callbacks of onVoice() are called from a separate thread.
    //
    private final VoiceRecorder.Callback mVoiceCallback = new VoiceRecorder.Callback() {
        @Override
        public void onVoiceStart() {
            deebug("onVoiceStart");
            if (mSpeechApiService == null) {
                deebug("mSpeechApiService is null onVoiceStart");
                return;
            }
            mSpeechApiService.startRecognizing(mVoiceRecorder.getSampleRate());
        }

        @Override
        public void onVoice(byte[] data, int size) {
            deebug("onVoice");
            if (mSpeechApiService == null) {
                deebug("mSpeechApiService is null onVoice");
                return;
            }
            mSpeechApiService.recognize(data, size);
        }

        @Override
        public void onVoiceEnd() {
            deebug("onVoiceEnd");
            if (mSpeechApiService != null) {
                deebug("mSpeechApiService is null onVoiceEnd");
                return;
            }
            mSpeechApiService.finishRecognizing();
        }

        @Override
        public void onStatusUpdate(String msg) {
            showStatus(msg);
        }
    };

    /**
     * Returns whether it has the permission, start the permission request flow asynchchronously
     * if it deosn't have the permission.
     * @param permissions
     * @param requestCode
     * @return true if it has the permission.
     */
    private boolean checkPermissions(String[] permissions, int requestCode) {
        List<String> permissionsMissing = new ArrayList<>();
        for (String p: permissions) {
            if (ContextCompat.checkSelfPermission(this, p)
                    != PackageManager.PERMISSION_GRANTED) {
                permissionsMissing.add(p);
            }
        }
        if (permissionsMissing.isEmpty()) {
            return true;
        }
        String msg = "";
        for (String p: permissionsMissing) {
            if (!msg.isEmpty()) {
                msg = msg = ", ";
            }
            msg += p;
        }
        Log.d("deebug", "requesting permission for " + msg);
        ActivityCompat.requestPermissions(this,
                (String[]) permissionsMissing.toArray(),
                requestCode);
        return false;
    }

    @Override
    public void onRequestPermissionsResult(
            int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        if (requestCode == PERMISSIONS_REQUEST_FOR_CLOUD) {
            for (int i = 0; i < grantResults.length; i++) {
                if (grantResults[i] != PackageManager.PERMISSION_GRANTED) {
                    Toast.makeText(this, "Missing permission: " + permissions[i],
                            Toast.LENGTH_SHORT).show();
                }
            }
        } else {
            super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        }
    }

    void showStatus(final CharSequence msg) {
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                mStatus.setText(msg);
            }
        });
    }

    private void deebug(final CharSequence msg) {
        Log.d("deebug", msg.toString());
//        Toast.makeText(getApplicationContext(), msg, Toast.LENGTH_SHORT).show();
    }
}
