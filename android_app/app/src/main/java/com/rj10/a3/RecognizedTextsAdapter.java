package com.rj10.a3;

import android.support.v7.widget.RecyclerView;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;
import android.widget.Toast;

import com.google.protobuf.Duration;

import java.util.List;

public class RecognizedTextsAdapter extends
        RecyclerView.Adapter<RecognizedTextsAdapter.MyViewHolder> {

    public interface ClickListener {
        void onClick(String waveFileName);
    }

    private List<RecognizedText> textList;
    private ClickListener listener;

    public RecognizedTextsAdapter(List<RecognizedText> textList, ClickListener listener) {
        this.textList = textList;
        this.listener = listener;
    }

    @Override
    public int getItemCount() {
        return textList.size();
    }

    @Override
    public MyViewHolder onCreateViewHolder(ViewGroup parent, int viewType) {
        View itemView = LayoutInflater.from(parent.getContext())
                .inflate(R.layout.speech_recog_row, parent, false);
        return new MyViewHolder(itemView, listener);
    }

    @Override
    public void onBindViewHolder(MyViewHolder myViewHolder, int position) {
        RecognizedText recogText = textList.get(position);
        myViewHolder.text.setText(recogText.getText());
    }

    public class MyViewHolder extends RecyclerView.ViewHolder implements View.OnClickListener {
        public TextView text;
        public ClickListener listener;

        public MyViewHolder(View view, ClickListener listener) {
            super(view);
            view.setOnClickListener(this);
            text = (TextView) view.findViewById(R.id.text);
            this.listener = listener;
        }

        @Override
        public void onClick(View v) {
            listener.onClick(text.getText().toString());
        }
    }
}
