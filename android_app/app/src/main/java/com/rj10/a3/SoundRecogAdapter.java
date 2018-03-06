package com.rj10.a3;

import android.content.Context;
import android.support.v7.widget.RecyclerView;
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
        void onItemClick(String waveFileName);
    }
    public void setOnItemClickListener(OnItemClickListener listener) {
        this.mItemClickListener = listener;
    }

    public static class ViewHolder extends RecyclerView.ViewHolder implements View.OnClickListener {
        public TextView itemStatus;
        public TextView wavFileTextView;
        public TextView soundLabelTextView;
        public TextView timestampTextView;
        public Button inferSoundTypeButton;

        public OnItemClickListener mClickListener;

        public ViewHolder(View itemView, OnItemClickListener clickListener) {
            super(itemView);
            itemView.setOnClickListener(this);
            mClickListener = clickListener;
            itemStatus = (TextView) itemView.findViewById(R.id.itemStatus);
            wavFileTextView = (TextView) itemView.findViewById(R.id.wavFileName);
            soundLabelTextView = (TextView) itemView.findViewById(R.id.soundLabel);
            timestampTextView = (TextView) itemView.findViewById(R.id.timestamp);
            inferSoundTypeButton = (Button) itemView.findViewById(R.id.inferSoundClass);
            inferSoundTypeButton.setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View view) {
                    soundLabelTextView.setText("infering...");
                }
            });
        }

        @Override
        public void onClick(View view) {
            mClickListener.onItemClick(wavFileTextView.getText().toString());
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
        if (item.mStatus != null) {
            viewHolder.itemStatus.setText(item.mStatus);
        }
        if (item.getWavFileName() != null) {
            viewHolder.wavFileTextView.setText(item.getWavFileName());
            viewHolder.inferSoundTypeButton.setVisibility(View.VISIBLE);
        } else {
            viewHolder.inferSoundTypeButton.setVisibility(View.GONE);
        }
        if (item.getLabel() != null) {
            viewHolder.soundLabelTextView.setText(item.getLabel());
        }
        if (item.getTimestamp() != null) {
            viewHolder.timestampTextView.setText(DATE_FORMAT.format(item.getTimestamp()));
        }
    }

    @Override
    public int getItemCount() {
        return mSoundRecogItems.size();
    }
}
