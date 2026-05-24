use serde::{Serialize, Deserialize};
use std::io::{Read, Write, Seek, SeekFrom};

pub const NEXA_MAGIC: [u8; 4] = *b"NEXA";
pub const NEXA_MAGIC_END: [u8; 4] = *b"AXEN";
const CURRENT_VERSION: u16 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NexaHeader {
    pub version: u16,
    pub flags: u16,
    pub dimension: u32,
    pub vector_count: u32,
    pub encoding_type: u8,
    pub checksum_algo: u8,
    pub metadata: String,
}

impl NexaHeader {
    pub fn new(dimension: u32, vector_count: u32) -> Self {
        NexaHeader {
            version: CURRENT_VERSION,
            flags: 0,
            dimension,
            vector_count,
            encoding_type: 0,
            checksum_algo: 1,
            metadata: String::new(),
        }
    }

    pub fn with_metadata(mut self, metadata: String) -> Self {
        self.metadata = metadata;
        self
    }
}

/// CRC32 checksum (simplified implementation)
pub fn crc32(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFFFFFF;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            if crc & 1 == 1 {
                crc = (crc >> 1) ^ 0xEDB88320;
            } else {
                crc >>= 1;
            }
        }
    }
    !crc
}

pub struct NexaFormat;

impl NexaFormat {
    /// Write a .nexa file with header, vector data, and checksums
    pub fn write<W: Write>(
        writer: &mut W,
        header: &NexaHeader,
        vectors: &[Vec<u8>],
    ) -> crate::error::Result<()> {
        // Magic
        writer.write_all(&NEXA_MAGIC)?;

        // Header fields
        writer.write_all(&header.version.to_le_bytes())?;
        writer.write_all(&header.flags.to_le_bytes())?;
        writer.write_all(&header.dimension.to_le_bytes())?;
        writer.write_all(&header.vector_count.to_le_bytes())?;
        writer.write_all(&[header.encoding_type])?;
        writer.write_all(&[header.checksum_algo])?;

        // Header checksum (of fields so far)
        let mut header_bytes = Vec::new();
        header_bytes.extend_from_slice(&header.version.to_le_bytes());
        header_bytes.extend_from_slice(&header.flags.to_le_bytes());
        header_bytes.extend_from_slice(&header.dimension.to_le_bytes());
        header_bytes.extend_from_slice(&header.vector_count.to_le_bytes());
        header_bytes.push(header.encoding_type);
        header_bytes.push(header.checksum_algo);
        let header_crc = crc32(&header_bytes);
        writer.write_all(&header_crc.to_le_bytes())?;

        // Metadata
        let meta_bytes = header.metadata.as_bytes();
        writer.write_all(&(meta_bytes.len() as u32).to_le_bytes())?;
        writer.write_all(meta_bytes)?;

        // Vector data with index tracking
        let mut offsets: Vec<(u32, u64, u32)> = Vec::new();
        let mut current_offset: u64 = 0;
        for (i, vec_data) in vectors.iter().enumerate() {
            offsets.push((i as u32, current_offset, vec_data.len() as u32));
            writer.write_all(vec_data)?;
            current_offset += vec_data.len() as u64;
        }

        // Index table offset placeholder position
        let index_table_marker = current_offset;

        // Index table
        let index_count = offsets.len() as u32;
        writer.write_all(&index_count.to_le_bytes())?;
        for (id, offset, length) in &offsets {
            writer.write_all(&id.to_le_bytes())?;
            writer.write_all(&offset.to_le_bytes())?;
            writer.write_all(&length.to_le_bytes())?;
        }

        // Footer checksum (of all vector data)
        let mut all_data = Vec::new();
        for v in vectors {
            all_data.extend_from_slice(v);
        }
        let footer_crc = crc32(&all_data);
        writer.write_all(&footer_crc.to_le_bytes())?;

        // End magic
        writer.write_all(&NEXA_MAGIC_END)?;

        Ok(())
    }

    /// Read a .nexa file, verifying checksums
    pub fn read<R: Read>(reader: &mut R) -> crate::error::Result<(NexaHeader, Vec<Vec<u8>>)> {
        // Magic
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if magic != NEXA_MAGIC {
            return Err(crate::NexaError::InvalidMagic);
        }

        // Header fields
        let mut buf2 = [0u8; 2];
        let mut buf4 = [0u8; 4];

        reader.read_exact(&mut buf2)?;
        let version = u16::from_le_bytes(buf2);
        if version > CURRENT_VERSION {
            return Err(crate::NexaError::UnsupportedVersion(version));
        }

        reader.read_exact(&mut buf2)?;
        let flags = u16::from_le_bytes(buf2);

        reader.read_exact(&mut buf4)?;
        let dimension = u32::from_le_bytes(buf4);

        reader.read_exact(&mut buf4)?;
        let vector_count = u32::from_le_bytes(buf4);

        let mut buf1 = [0u8; 1];
        reader.read_exact(&mut buf1)?;
        let encoding_type = buf1[0];

        reader.read_exact(&mut buf1)?;
        let checksum_algo = buf1[0];

        // Verify header checksum
        reader.read_exact(&mut buf4)?;
        let stored_header_crc = u32::from_le_bytes(buf4);

        let mut header_bytes = Vec::new();
        header_bytes.extend_from_slice(&version.to_le_bytes());
        header_bytes.extend_from_slice(&flags.to_le_bytes());
        header_bytes.extend_from_slice(&dimension.to_le_bytes());
        header_bytes.extend_from_slice(&vector_count.to_le_bytes());
        header_bytes.push(encoding_type);
        header_bytes.push(checksum_algo);
        let computed_crc = crc32(&header_bytes);
        if stored_header_crc != computed_crc {
            return Err(crate::NexaError::ChecksumMismatch {
                expected: stored_header_crc,
                got: computed_crc,
            });
        }

        // Metadata
        reader.read_exact(&mut buf4)?;
        let meta_len = u32::from_le_bytes(buf4) as usize;
        let mut meta_bytes = vec![0u8; meta_len];
        reader.read_exact(&mut meta_bytes)?;
        let metadata = String::from_utf8(meta_bytes)
            .map_err(|e| crate::NexaError::FormatError(format!("invalid metadata UTF-8: {}", e)))?;

        // Read remaining data (vectors + index + footer)
        let mut remaining = Vec::new();
        reader.read_to_end(&mut remaining)?;

        // Parse from end: magic_end (4) + footer_crc (4) = 8 bytes from end
        if remaining.len() < 8 {
            return Err(crate::NexaError::FormatError("truncated file".into()));
        }

        let end_magic_start = remaining.len() - 4;
        let end_magic: [u8; 4] = remaining[end_magic_start..].try_into().unwrap();
        if end_magic != NEXA_MAGIC_END {
            return Err(crate::NexaError::InvalidMagic);
        }

        let footer_crc_start = end_magic_start - 4;
        let stored_footer_crc = u32::from_le_bytes(
            remaining[footer_crc_start..footer_crc_start + 4].try_into().unwrap()
        );

        // Parse index table (just before footer_crc)
        // Index table: count(4) + count * (id(4) + offset(8) + length(4))
        // We need to find where index table starts
        // Read index_count from before the entries
        // Work backwards from footer_crc_start

        // First, figure out index entry count by reading at the right position
        // The index table is: [index_count: u32] [entries: (u32, u64, u32) * count]
        // Total index size = 4 + count * 16
        // But we don't know count yet...

        // Alternative: parse forward through vector data using index
        // We know vector_count, so we can read the index_count after vector data

        // Strategy: the vector data comes first, then the index table, then footer crc, then end magic
        // Let's find index table by scanning for index_count at possible positions

        // Actually, simpler: read vector data sizes from index table
        // The index table start position = footer_crc_start - (4 + vector_count * 16)
        let index_entry_size = 16usize; // u32 + u64 + u32
        let index_table_size = 4 + (vector_count as usize) * index_entry_size;

        if remaining.len() < 8 + index_table_size {
            return Err(crate::NexaError::FormatError("truncated index table".into()));
        }

        let index_start = footer_crc_start - index_table_size;

        let idx_count = u32::from_le_bytes(
            remaining[index_start..index_start + 4].try_into().unwrap()
        );

        let mut vectors = Vec::new();
        let vector_data = &remaining[..index_start];

        let mut pos = index_start + 4;
        for _ in 0..idx_count {
            let _id = u32::from_le_bytes(remaining[pos..pos + 4].try_into().unwrap());
            pos += 4;
            let offset = u64::from_le_bytes(remaining[pos..pos + 8].try_into().unwrap()) as usize;
            pos += 8;
            let length = u32::from_le_bytes(remaining[pos..pos + 4].try_into().unwrap()) as usize;
            pos += 4;

            if offset + length > vector_data.len() {
                return Err(crate::NexaError::FormatError("vector data out of bounds".into()));
            }
            vectors.push(vector_data[offset..offset + length].to_vec());
        }

        // Verify footer checksum
        let computed_footer_crc = crc32(vector_data);
        if stored_footer_crc != computed_footer_crc {
            return Err(crate::NexaError::ChecksumMismatch {
                expected: stored_footer_crc,
                got: computed_footer_crc,
            });
        }

        let header = NexaHeader {
            version,
            flags,
            dimension,
            vector_count: idx_count,
            encoding_type,
            checksum_algo,
            metadata,
        };

        Ok((header, vectors))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn nexa_format_write_then_read_roundtrip() {
        let header = NexaHeader::new(10000, 3)
            .with_metadata(r#"{"type":"test"}"#.to_string());

        let vectors = vec![
            vec![1u8, 2, 3, 4, 5],
            vec![10, 20, 30],
            vec![100, 200],
        ];

        let mut buf = Vec::new();
        NexaFormat::write(&mut buf, &header, &vectors).unwrap();

        let mut cursor = Cursor::new(&buf);
        let (read_header, read_vectors) = NexaFormat::read(&mut cursor).unwrap();

        assert_eq!(read_header.dimension, 10000);
        assert_eq!(read_header.vector_count, 3);
        assert_eq!(read_header.metadata, r#"{"type":"test"}"#);
        assert_eq!(read_vectors, vectors);
    }

    #[test]
    fn nexa_format_detects_invalid_magic() {
        let data = b"NOPE\x01\x00";
        let mut cursor = Cursor::new(data.as_slice());
        assert!(matches!(NexaFormat::read(&mut cursor), Err(crate::NexaError::InvalidMagic)));
    }

    #[test]
    fn nexa_format_detects_corrupted_header() {
        let header = NexaHeader::new(10000, 1);
        let vectors = vec![vec![1u8, 2, 3]];
        let mut buf = Vec::new();
        NexaFormat::write(&mut buf, &header, &vectors).unwrap();

        // Corrupt a header byte
        buf[6] ^= 0xFF;

        let mut cursor = Cursor::new(&buf);
        assert!(matches!(NexaFormat::read(&mut cursor), Err(crate::NexaError::ChecksumMismatch { .. })));
    }

    #[test]
    fn crc32_deterministic() {
        let data = b"hello nexacore";
        let c1 = crc32(data);
        let c2 = crc32(data);
        assert_eq!(c1, c2);
        assert_ne!(c1, 0);
    }
}
