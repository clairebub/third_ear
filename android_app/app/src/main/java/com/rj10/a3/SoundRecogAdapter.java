package com.rj10.a3;

import android.content.Context;
import android.graphics.Color;
import android.support.v7.widget.RecyclerView;
import android.text.TextUtils;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.TextView;

import org.w3c.dom.Text;

import java.text.SimpleDateFormat;


import java.util.List;

/**
 * Created by tangfamily on 3/5/18.
 */

public class SoundRecogAdapter extends RecyclerView.Adapter<SoundRecogAdapter.ViewHolder> {
    // Define the listener interface
    private OnItemClickListener mItemClickListener;
    public interface OnItemClickListener {
        void onItemClick(SoundRecogItem item);
    }
    public void setOnItemClickListener(OnItemClickListener listener) {
        this.mItemClickListener = listener;
    }

    public static class ViewHolder extends RecyclerView.ViewHolder implements View.OnClickListener {
        public SoundRecogItem item;
        public TextView timestampTextView;
        public TextView textRecognized;
        public TextView wavFileRecorded;
        public TextView soundLabel;
        public TextView soundLabelLabel;
        public OnItemClickListener mClickListener;

        public ViewHolder(View itemView, OnItemClickListener clickListener) {
            super(itemView);
            itemView.setOnClickListener(this);
            mClickListener = clickListener;

            timestampTextView = (TextView) itemView.findViewById(R.id.timestamp);
            textRecognized = (TextView) itemView.findViewById(R.id.textRecognized);
            wavFileRecorded = (TextView) itemView.findViewById(R.id.wavFileRecorded);
            soundLabel = (TextView) itemView.findViewById(R.id.soundLabel);
            soundLabelLabel = itemView.findViewById(R.id.soundLabelLabel);
        }

        @Override
        public void onClick(View view) {
            if (wavFileRecorded.getVisibility() == View.VISIBLE && !wavFileRecorded.getText().toString().isEmpty()) {
                mClickListener.onItemClick(item);
            }
        }
    }

    private Context mContext;
    private List<SoundRecogItem> mSoundRecogItems;
    private static final SimpleDateFormat DATE_FORMAT = new SimpleDateFormat("yyyy-mm-dd hh:mm:ss");


    public SoundRecogAdapter(Context context, List<SoundRecogItem> soundRecogItems) {
        mContext = context;
        mSoundRecogItems = soundRecogItems;
    }

    @Override
    public ViewHolder onCreateViewHolder(ViewGroup parent, int viewType) {
        Context context = parent.getContext();
        LayoutInflater inflater = LayoutInflater.from(context);

        View row = inflater.inflate(R.layout.sound_recog_row, parent, false );
        return new ViewHolder(row, mItemClickListener);
    }

    @Override
    public void onBindViewHolder(ViewHolder viewHolder, int position) {
        SoundRecogItem item = mSoundRecogItems.get(position);
        viewHolder.item = item;
        if (!TextUtils.isEmpty(item.wavFileRecorded)) {
            String[] path = TextUtils.split(item.wavFileRecorded, "/");
            Log.d("DEEBUG", "wavFilePath=" + item.wavFileRecorded);
            for(String p : path) {
                Log.d("DEEBUG", "path=" + p);
            }
            String waveFileName = path[path.length-1];
            viewHolder.wavFileRecorded.setVisibility(View.VISIBLE);
            viewHolder.soundLabelLabel.setVisibility(View.VISIBLE);
            viewHolder.soundLabel.setVisibility(View.VISIBLE);
            viewHolder.soundLabel.setText(item.wavSoundType);
            viewHolder.soundLabel.setTextColor(Color.BLUE);
            viewHolder.wavFileRecorded.setText(waveFileName);
            viewHolder.textRecognized.setVisibility(View.GONE);
        }
        viewHolder.timestampTextView.setText(DATE_FORMAT.format(item.timestamp));
        if (item.textRecognized != null) {
            viewHolder.textRecognized.setVisibility(View.VISIBLE);
            viewHolder.textRecognized.setText(item.textRecognized);
            viewHolder.wavFileRecorded.setVisibility(View.GONE);
            viewHolder.soundLabel.setVisibility(View.GONE);
            viewHolder.soundLabelLabel.setVisibility(View.GONE);
        }
    }

    @Override
    public int getItemCount() {
        return mSoundRecogItems.size();
    }
}
