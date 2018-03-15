package com.rj10.a3;

import android.Manifest;
import android.content.ComponentName;
import android.content.Context;
import android.content.Intent;
import android.content.ServiceConnection;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.media.MediaPlayer;
import android.os.IBinder;
import android.os.Looper;
import android.support.annotation.NonNull;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.support.v7.widget.DividerItemDecoration;
import android.support.v7.widget.LinearLayoutManager;
import android.support.v7.widget.RecyclerView;
import android.text.TextUtils;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.Toast;


import java.io.IOException;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;

import be.tarsos.dsp.AudioDispatcher;
import be.tarsos.dsp.AudioEvent;
import be.tarsos.dsp.AudioProcessor;
import be.tarsos.dsp.io.android.AudioDispatcherFactory;
import be.tarsos.dsp.pitch.PitchDetectionHandler;
import be.tarsos.dsp.pitch.PitchDetectionResult;
import be.tarsos.dsp.pitch.PitchProcessor;

import org.tensorflow.Operation;
import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

//
// An app that monitors the ambient sound and do speech detection and outputs the results
// on the screen.
// The UI is the following:
// 1. a button to start/stop the speech detection and indicate whether it's running
// 2. an output area to display the detected speech
//
public class MainActivity extends AppCompatActivity
        implements ActivityCompat.OnRequestPermissionsResultCallback {

    private static final String TAG = "3rdEarMainActivity";
    private final int PERMISSIONS_REQUEST_FOR_CLOUD = 1;

    private final int REQ_CODE_SPEECH_RECOG_LOCAL = 100;

    private Button mStartButton;
    private Button mStopButton;
    private RecyclerView mRVSounds;
    private List<SoundRecogItem> mSoundItems = new ArrayList<>();
    private SoundRecogAdapter mSoundRecogAdapter;
    private CheckBox mConversationMode;

    private SoundRecorder mVoiceRecorder;
    private SpeechApiService mSpeechApiService;

    private TensorFlowInferenceInterface inferenceInterface;
    private Operation operation;

    private static final String MODEL_FILE = "file:///android_asset/img/frozen_vear_model.pb";
    private static final String INPUT_NAME = "X";
    private static final String OUTPUT_NAME = "pred";
    private static final String[] SOUND_CLASSES = {
            "air_conditioner",
            "car_horn",
            "children_playing",
            "dog_bark",
            "drilling",
            "engine_idling",
            "gun_shot",
            "jackhammer",
            "siren",
            "street_music",
    };

    /** Defines callbacks for service binding, passed to bindService() */
    private final ServiceConnection mSpeechApiServiceConnection = new ServiceConnection() {
        @Override
        public void onServiceConnected(ComponentName className,
                                       IBinder service) {
            // We've bound to LocalService, cast the IBinder and get LocalService instance
            SpeechApiService.SpeechApiBinder binder = (SpeechApiService.SpeechApiBinder) service;
            mSpeechApiService = binder.getService();
            mSpeechApiService.setCallback(mSpeechApiServiceCallback);
            Log.d("SpeechApiService","onServiceConnected(): " + mSpeechApiService);
        }

        @Override
        public void onServiceDisconnected(ComponentName arg0) {
            mSpeechApiService = null;
            Log.d("SpeechApiService","onServiceDisconnected()" );
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        Log.d(TAG, "onCreate");

        setContentView(R.layout.activity_main);

        mConversationMode = (CheckBox) findViewById(R.id.conversationMode);
        mConversationMode.setChecked(false);

        mStartButton = (Button) findViewById(R.id.startButton);
        mStartButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                handleStartButtonClick((Button) v);
            }
        });
        mStartButton.setEnabled(true);

        mStopButton = (Button) findViewById(R.id.stopButton);
        mStopButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                handleStopButtonClick((Button) v);
            }
        });
        mStopButton.setEnabled(false);

        Button clearButton = (Button) findViewById(R.id.clearButton);
        clearButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                mSoundItems.clear();
                mSoundItems.add(SoundRecogItem.createItemForTextRecognized(new Date(), "Virtual Ear is ready to use."));
                mSoundRecogAdapter.notifyDataSetChanged();
            }
        });

        mRVSounds = (RecyclerView) findViewById(R.id.soundsRecyclerView);
        RecyclerView.LayoutManager layoutManager = new LinearLayoutManager(this);
        mRVSounds.setLayoutManager(layoutManager);
        RecyclerView.ItemDecoration itemDecoration = new
                DividerItemDecoration(this, DividerItemDecoration.VERTICAL);
        mRVSounds.addItemDecoration(itemDecoration);

        mSoundRecogAdapter = new SoundRecogAdapter(this, mSoundItems);
        mSoundRecogAdapter.setOnItemClickListener(new SoundItemClickListener(this));
        mRVSounds.setAdapter(mSoundRecogAdapter);

        mSoundItems.add(SoundRecogItem.createItemForTextRecognized(new Date(), "Virtual Ear is ready to use."));
        mSoundRecogAdapter.notifyDataSetChanged();

        final AssetManager assetManager = this.getAssets();
        Log.d(TAG, "checking assets");
        try {
            String[] pList = assetManager.list("img");
            for (String p: pList) {
                Log.d(TAG, "deebug p=" + p);
            }
        } catch (IOException ex) {
            Log.d(TAG, ex.getMessage());
        }

        inferenceInterface = new TensorFlowInferenceInterface(assetManager, MODEL_FILE);
        operation = inferenceInterface.graphOperation(OUTPUT_NAME);
    }

    private String inferSoundClass(final float[] mfcc) {
        final int numClasses = (int) operation.output(0).shape().size(1);
        Log.d(TAG, "numClasses=" + numClasses);
        float[] yy = new float[10];
        for (int i = 0; i < 10; i++) {
            yy[i] = 0;
        }
        inferenceInterface.feed(INPUT_NAME, mfcc, 1, 40);
        boolean logStats = false;
        String[] outputNames = new String[] {OUTPUT_NAME};
        inferenceInterface.run(outputNames, logStats);
        inferenceInterface.fetch(OUTPUT_NAME, yy);
        StringBuffer sb = new StringBuffer("pred: ");
        for(int i = 0; i < yy.length; i++) {
            sb.append(yy[i] + " ");
        }
        sb.append("\n");
        Log.d(TAG, sb.toString());
        int classIndex = -1;
        for (int i = 0; i < 10; i++) {
            if (classIndex < 0 || yy[i] > yy[classIndex]) {
                classIndex = i;
            }
        }
        return SOUND_CLASSES[classIndex];
    }

    /**
     * Bind to the speech api service only when the activity is in use. You can do it at OnStart()
     * or at onResume()
     */
    @Override
    protected void onStart() {
        super.onStart();
        Log.d(TAG, "onStart");
        // Prepare Cloud Speech API
        bindService(new Intent(this, SpeechApiService.class),
                mSpeechApiServiceConnection, BIND_AUTO_CREATE);
    }

    @Override
    protected void onResume() {
        super.onResume();
        Log.d(TAG, "onResume");
    }

    @Override
    protected void onPause() {
        super.onPause();
        Log.d(TAG, "onPause");
    }

    @Override
    protected void onStop() {
        super.onStop();
        Log.d(TAG, "onStop");
        unbindService(mSpeechApiServiceConnection);
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        Log.d(TAG, "onDestroy");
    }

    private void handleStartButtonClick(Button button) {
        mStartButton.setEnabled(false);
        mStopButton.setEnabled(true);
        startSpeechRec();
    }

    private void handleStopButtonClick(Button button) {
        mStopButton.setEnabled(false);
        mStartButton.setEnabled(true);
        stopSpeechRec();
    }

    private void startSpeechRec() {
        String[] permissions = new String[] {
                Manifest.permission.RECORD_AUDIO,
                Manifest.permission.READ_PHONE_STATE,
                Manifest.permission.WRITE_EXTERNAL_STORAGE
        };
        if (!checkPermissions(permissions, PERMISSIONS_REQUEST_FOR_CLOUD)) {
            return;
        }

        if (mVoiceRecorder != null) {
            mVoiceRecorder.stop();
        }
        mVoiceRecorder = new SoundRecorder();
        mVoiceRecorder.start(mVoiceCallback);
    }


    private void stopSpeechRec() {
        Log.d(TAG, "stopSpeechRecStreaming: " + mVoiceRecorder);
        if (mVoiceRecorder != null) {
            mVoiceRecorder.stop();
            mVoiceRecorder = null;
        }
    }

    private final SpeechApiService.Callback mSpeechApiServiceCallback =
            new SpeechApiService.Callback() {
        private String lastTextRecognized = null;
        @Override
        public void onSpeechRecognized(final String text, final boolean isFinal) {
            if (isFinal) {
                // TODO:
            }
            if (text != null && !TextUtils.isEmpty(text)) {
                Log.d("DEEBUG", "speed recog: " + text);
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        //if(isFinal) {
                        if(lastTextRecognized == null || !text.equalsIgnoreCase(lastTextRecognized)) {
                            mSoundItems.add(SoundRecogItem.createItemForTextRecognized(new Date(), text));
                            Log.d(TAG, "isFinal=" + isFinal + ", got text: " + text);
                            mSoundRecogAdapter.notifyDataSetChanged();
                            lastTextRecognized = text;
                        }
                        //}
                        mRVSounds.scrollToPosition(mSoundItems.size()-1);
                    }
                });
            }
        }
    };

    //
    // Callback from voice recorder to handle start/ongoing/end of voice input.
    // For cloud based speech recog streaming Api only.
    // The callbacks of onVoice() are called from a separate thread.
    //
    private final SoundRecorder.Callback mVoiceCallback = new SoundRecorder.Callback() {
        private int mSampleRate = -1;
        @Override
        public void onVoiceStart(int sampleRate) {
            if (!mConversationMode.isChecked()) {
                return;
            }
            mSampleRate = sampleRate;
            Log.d(TAG, "SoundRecorder.onVoiceStart(): " + (Looper.myLooper() == Looper.getMainLooper()));
            if (mSpeechApiService == null) {
                Log.d(TAG, "mSpeechApiService is null onVoiceStart");
                return;
            }
            mSpeechApiService.startRecognizing(sampleRate);
        }

        @Override
        public void onVoice(byte[] data, int size) {
            if (!mConversationMode.isChecked()) {
                return;
            }
            // Log.d(TAG,"SoundRecorder.onVoice(): " + (Looper.myLooper() == Looper.getMainLooper()));
            if (mSpeechApiService == null) {
                Log.d(TAG, "mSpeechApiService is null onVoice");
                return;
            }
            if (mSampleRate < 0) {
                Log.e(TAG, "sampleRate is -1 when onVoice");
                return;
            }
            mSpeechApiService.recognize(data, size);
        }

        @Override
        public void onVoiceEnd() {
            Log.d(TAG,"SoundRecorder.onVoiceEnd(): " + (Looper.myLooper() == Looper.getMainLooper()));
            if (!mConversationMode.isChecked()) {
                return;
            }
            if (mSpeechApiService != null) {
                Log.d(TAG,"mSpeechApiService is null onVoiceEnd");
                return;
            }
            mSpeechApiService.finishRecognizing();
        }

        @Override
        public void deebug(String msg) {
            Log.d(TAG,"SoundRecorder.deebug(): " + msg + ", " + (Looper.myLooper() == Looper.getMainLooper()));
            //showStatus(msg);
        }

        @Override
        public void onWaveFilePublished(final String waveFilePath, final float[] mfcc) {
            // TODO: run inference
            final String soundClass = inferSoundClass(mfcc);
            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    if (mConversationMode.isChecked()) {
                        return;
                    }
                    SoundRecogItem item = SoundRecogItem.createItemForWavFileRecorded(new Date(), waveFilePath, soundClass);
                    mSoundItems.add(item);
                    mSoundRecogAdapter.notifyDataSetChanged();
                    mRVSounds.scrollToPosition(mSoundItems.size()-1);
                }
            });
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
        Log.d(TAG, "requesting permission for " + msg);
        String[] p2 = new String[permissionsMissing.size()];
        for(int i = 0; i < permissionsMissing.size(); i++) {
            p2[i] = permissionsMissing.get(i);
        }
        ActivityCompat.requestPermissions(this,
                p2,
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

    private void tryTarsos() {
        AudioDispatcher dispatcher = AudioDispatcherFactory.fromDefaultMicrophone(22050,1024,0);

        PitchDetectionHandler pdh = new PitchDetectionHandler() {
            @Override
            public void handlePitch(PitchDetectionResult result, AudioEvent e) {
                final float pitchInHz = result.getPitch();
                runOnUiThread(new Runnable() {
                    @Override
                    public void run() {
                        //TextView text = (TextView) findViewById(R.id.appStatus);
                        //text.setText("" + pitchInHz);
                    }
                });
            }
        };
        AudioProcessor p = new PitchProcessor(PitchProcessor.PitchEstimationAlgorithm.FFT_YIN, 22050, 1024, pdh);
        dispatcher.addAudioProcessor(p);
        new Thread(dispatcher,"Audio Dispatcher").start();
    }

    private void playWavFile() {
        MediaPlayer mediaPlayer = MediaPlayer.create(getApplicationContext(), R.raw.applause_y);
        mediaPlayer.start();
    }

    class SoundItemClickListener implements SoundRecogAdapter.OnItemClickListener {
        private final Context mContext;
        private MediaPlayer mMediaPlayer;

        public SoundItemClickListener(Context context) {
            mContext = context;
        }

        @Override
        public void onItemClick(SoundRecogItem item) {
            String waveFileName = item.wavFileRecorded;
            Log.w(TAG, "clicked file " + waveFileName);
            try {
                if (mMediaPlayer != null) {
                    mMediaPlayer.release();
                    mMediaPlayer = null;
                }
                mMediaPlayer = new MediaPlayer();
                mMediaPlayer.setDataSource(waveFileName);
                mMediaPlayer.setOnPreparedListener(new MediaPlayer.OnPreparedListener() {
                    @Override
                    public void onPrepared(MediaPlayer player) {
                        player.start();
                    }
                });
                mMediaPlayer.prepareAsync();
            } catch (IOException ex) {
                Toast.makeText(mContext, waveFileName + " not found", Toast.LENGTH_SHORT).show();
                Log.e(TAG, ex.toString());
            }
        }
    }
}
